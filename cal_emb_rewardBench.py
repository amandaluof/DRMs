# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys
import glob
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import Any, Dict, List, Optional, Union, Tuple
import transformers
from transformers.cache_utils import Cache
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from rewardbench import (
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    load_eval_dataset,
    save_to_hub,
    torch_dtype_mapping,
)
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section
from rewardbench.models.pipeline import RewardBenchPipeline

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer to model")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length of RM inputs (passed to pipeline)")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--debug", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument(
        "--not_quantized", action="store_true", help="disable quantization for models that are quantized by default"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "float64"],
        help="PyTorch dtype (default: float16)",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation to use (default: None)",
    )
    parser.add_argument(
        "--deep_inf", action="store_true", help="use intermediate feats to inference"
    )
    parser.add_argument(
        "--num_expert", type=int, help="number of feats to use"
    )
    parser.add_argument(
        "--expert_idx", type=int, default=-1, help="index of intermediate feats"
    )
    parser.add_argument(
        "--normalize", action="store_true", help="use intermediate feats to inference"
    )
    parser.add_argument(
        "--normalize_statistics", type=str, help="path to moe score statictics"
    )
    parser.add_argument(
        "--rm_head", type=str, help="path to moe score statictics",  default=None
    )
    parser.add_argument(
        "--emb_path", type=str, help="path to moe score statictics", default=""
    )
    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
    return args

class EmbRewardBenchClass(RewardBenchPipeline):
    def __init__(self, task, model, tokenizer):
        super().__init__(task=task, model=model, tokenizer=tokenizer)

    def __call__(self, samples, return_inputs=False, **kwargs):
        _ = kwargs.get("batch_size", 1)
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", 2048)
        inputs = self.tokenizer(
            samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            # return_special_tokens_mask=True,
            return_tensors="pt",
        ).to("cuda")

        # if tokenizer.bos_token exists, check if there is a double bos token to start the inputs
        # if so, we'll remove the first one and pass in the inputs (somewhat hacky solution)
        # a full refactor can be done to use tokenizer.apply_chat_template(chat, tokenize=True)
        # though, so many RM implementations are non standard, so this is a quick fix rather than ecosystem wide
        if self.tokenizer.bos_token:
            bos_token_id = self.tokenizer.bos_token_id
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Ensure input_ids is 2D
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

            # Find the start of each sequence (first non-pad token)
            seq_starts = attention_mask.argmax(dim=1)

            # Check for double BOS tokens
            seq_second = torch.clamp(seq_starts + 1, max=input_ids.size(1) - 1)
            double_bos_mask = (input_ids[torch.arange(input_ids.size(0)), seq_starts] == bos_token_id) & (
                input_ids[torch.arange(input_ids.size(0)), seq_second] == bos_token_id
            )

            # Set attention mask to 0 for the first BOS token where double BOS is detected
            if double_bos_mask.any():
                attention_mask[
                    torch.arange(attention_mask.size(0), device=attention_mask.device)[double_bos_mask],
                    seq_starts[double_bos_mask],
                ] = torch.tensor(0, device=attention_mask.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        if return_inputs:
            return outputs, inputs
        else:
            return outputs


def main():
    args = get_args()
    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()
    current_device = accelerator.process_index

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    if args.model in REWARD_MODEL_CONFIG:
        config = REWARD_MODEL_CONFIG[args.model]
    else:
        config = REWARD_MODEL_CONFIG["default"]
    logger.info(f"Using reward model config: {config}")

    # Default entries
    # "model_builder": AutoModelForSequenceClassification.from_pretrained,
    # "pipeline_builder": pipeline,
    # "quantized": True,
    # "custom_dialogue": False,
    # "model_type": "Seq. Classifier"

    quantized = config["quantized"]  # only Starling isn't quantized for now
    # if llama-3 in name, switch quantized to False (severely degrades performance)
    if (
        ("llama-3" in args.model)
        or ("Llama3" in args.model)
        or ("Llama-3" in args.model)
        or ("LLaMA3" in args.model)
        or ("llama3" in args.model)
        or args.not_quantized
    ):
        quantized = False
        logger.info(f"Disabling quantization for llama-3 or override flag (--not_quantized: {args.not_quantized})")

    custom_dialogue = config["custom_dialogue"]
    model_type = config["model_type"]
    model_builder = config["model_builder"]
    pipeline_builder = config["pipeline_builder"]
    torch_dtype = config.get("torch_dtype", None)
    # if not datatype in config (default), check args
    if torch_dtype is None:
        # if datatype is bfloat16, then manually turn off quantizaiton (done with bitsandbytes)
        if args.torch_dtype == torch.bfloat16:
            quantized = False
            logger.info("Disabling quantization for bfloat16 datatype")
        torch_dtype = args.torch_dtype

    # not included in config to make user explicitly understand they are passing this
    trust_remote_code = args.trust_remote_code

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if not custom_dialogue:  # not needed for PairRM / SteamSHP
        tokenizer.truncation_side = "left"  # copied from Starling, but few samples are above context length
    
    # ### add by amandaa
    if (("llama-3" in args.model) or ("Llama3" in args.model) or ("Llama-3" in args.model) or ("LLaMA3" in args.model) or ("llama3" in args.model)) and "-Instruct-" in args.model:
        # import pdb;pdb.set_trace()
        # if tokenizer.pad_token_id is None:        
        #     tokenizer.pad_token_id = 128256
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=conv,
        custom_dialogue_formatting=custom_dialogue,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id"],
    )
    # copy id for saving, then remove
    ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]
        ids = ids[:10]

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
    assert BATCH_SIZE == 1
    logger.info("*** Load reward model ***")
    reward_pipeline_kwargs = {
        "batch_size": BATCH_SIZE,  # eval_args.inference_batch_size,
        "truncation": True,
        "padding": True,
        "max_length": args.max_length,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }
    if quantized:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": {"": current_device},
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch_dtype,
        }

    # if attn_implementation is not specified, this falls back to Hugging Face's default
    # strategy (which chooses between sdpa and eager depending on pytorch version)
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = model_builder(args.model, **model_kwargs, trust_remote_code=trust_remote_code)
    model.resize_token_embeddings(len(tokenizer)) ### add by amandaa

    if args.rm_head is not None:
        info = model.score.load_state_dict(torch.load(args.rm_head))
        print("replace the reward model head")
        print(info)

    def custom_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0] # [2, 497, 2048]
        scores = self.score(hidden_states) # [N, 5]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(scores.device)
            else:
                sequence_lengths = -1

        emb = hidden_states[:, sequence_lengths[0], :]
        
        loss = None   
        weighted_scores = None 
        
        if not return_dict:
            output = (weighted_scores,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=weighted_scores,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=emb,
            attentions=transformer_outputs.attentions,
        )
    
    # hack model's forward
    model.original_forward = model.forward
    model.forward = custom_forward.__get__(model, type(model))
    
    reward_pipe = EmbRewardBenchClass(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
    )

    ############################
    # Tokenization settings & dataset preparation
    ############################
    # set pad token to eos token if not set
    print("reward_pipe.tokenizer.pad_token_id:", reward_pipe.tokenizer.pad_token_id)
    print("reward_pipe.tokenizer.eos_token_id:", reward_pipe.tokenizer.eos_token_id)
    if reward_pipe.tokenizer.pad_token_id is None: # adapt to LLama
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    # For models whose config did not contains `pad_token_id`
    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

    # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True

    ############################
    # Run inference [1/2]" built in transformers
    ############################
    # if using HF pipeline, can pass entire dataset and get results
    # first, handle custom pipelines that we must batch normally
    if pipeline_builder == pipeline:
        NotImplementedError
    else:
        logger.info("*** Running dataloader to collect results ***")
        # TODO make more custom pipelines work with pre-tokenized data
        from torch.utils.data.dataloader import default_collate

        # for PairRM, hmm, will move all of this later
        def custom_collate_fn(batch):
            # check if ['text_chosen'] is in first batch element
            # Check if the first element of the batch is a dictionary
            if isinstance(batch[0]["text_chosen"][0], dict):
                return batch  # Return the batch as-is if it's a list of dicts
            else:
                return default_collate(batch)  # Use the default collate behavior otherwise

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=custom_collate_fn,  # if not args.pref_sets else None,
            shuffle=False,
            drop_last=False,
        )

        dataloader, model = accelerator.prepare(dataloader, reward_pipe.model)
        reward_pipe.model = model

        results = []
        embs = []
        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            # logger.info(f"RM inference step {step}/{len(dataloader)}")

            if model_type == "Custom Classifier":
                NotImplementedError
            else:
                output_chosen = reward_pipe(batch["text_chosen"], **reward_pipeline_kwargs) # (N, 1)
                output_rejected = reward_pipe(batch["text_rejected"], **reward_pipeline_kwargs)
                emb_chosen = output_chosen.hidden_states
                emb_rejected = output_rejected.hidden_states
                emb = torch.cat([emb_chosen, emb_rejected], 0)[None,...] # (2， 2048)
                embs.append(emb)


    # save results
    embs = torch.cat(embs, 0) # (N, 2, 2048)
    # out_dataset = out_dataset.add_column("subset", subsets)
    # out_dataset = out_dataset.add_column("id", ids)
    data = {"embs": embs, "subsets":subsets, "ids": ids}
    print("embs:", embs.shape)
    print("subsets:", len(subsets))
    print("ids:", len(ids))

    model_arch = args.model.split("/")[1]
    print("model_arch:", model_arch)
    
    os.makedirs(args.emb_path, exist_ok=True)

    with open(f"{args.emb_path}/rewardBench_{model_arch}.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
