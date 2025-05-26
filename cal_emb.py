from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator
import evaluate
import numpy as np
import os
import tqdm
import pickle
import random
import pandas as pd
from typing import Tuple
from collections import defaultdict
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

from trl import RewardTrainer
from trl.trainer.utils import (
    decode_and_strip_padding,
    print_rich_table,
)
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from accelerate.utils import gather_object
from transformers.utils import PaddingStrategy
from transformers.trainer_pt_utils import nested_detach
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["HF_TOKEN"] = 'xxxxxxxxxxxx'


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_train_batch_size: Optional[int] = field(default=1) 
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=16)
    learning_rate: Optional[float] = field(default=5e-6)
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "The lr scheduler"},)
    max_length: Optional[int] = field(default=1024) # 1024 512 , 600
    use_lora: Optional[bool] = field(default=False)
    base_model: Optional[str] = field(default='')
    wandb_name: Optional[str] = field(default="mistral-7b-it_reward_unified_0.5dataset_lora32_len1024_2poch_marginfix",)
    log_dir: Optional[str] = field(default='./reward_models_unified_dpo_new')
    loss_type: Optional[str] = field(default='origin')
    use_smallset: Optional[bool] = field(default=False)
    freeze_pretrained: Optional[bool] = field(default=False)
    data_path: Optional[str] = field(default='weqweasdas/preference_dataset_mixture2_and_safe_pku')
    save_steps: Optional[int] = field(default=100)
    cls_embs_path: Optional[str] = field(default='')
    debug: Optional[bool] = field(default=False)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model_name = script_args.base_model
tokenizer_name = model_name
data_path = script_args.data_path

def check(c_messages, r_messages):
    if len(c_messages) == len(r_messages):
        for idx, (c_item, r_itme) in enumerate(zip(c_messages, r_messages)):
            if idx == len(c_messages) - 1 and c_item['role']== "assistant" and r_itme['role'] == "assistant":
                return True
            if c_item['role'] == r_itme['role'] and c_item['content'] == r_itme['content']:
                pass
            else:
                return False
    else:
        return False

token_patterns = {
    # Llama3 token IDs of "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "llama3": [128009, 128006, 78191, 128007, 271],
    # Gemma2 token IDs of "<end_of_turn>\n<start_of_turn>model\n"
    "gemma2": [107, 108, 106, 2516, 108],
}


def find_token_for_gating(lst, model_family):
    """Find the last occurrence of a token_pattern in a list."""
    token_pattern = token_patterns[model_family]
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    for j in range(search_end - token_pattern_len, -1, -1):
        if lst[j : j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")

def build_dataset_mix(data_path, tokenizer, split='train', size=None):
    ds = load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(0, size))

    # add num_index for dataset
    new_column = list(range(len(ds)))
    ds = ds.add_column("data_index", new_column)
    print("lenghth of dataset:", len(ds))
    
    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"padding": 'max_length', "truncation": True, "max_length": script_args.max_length, "return_tensors": "pt"}
        # kwargs = {"return_tensors": "pt"}
        chosen_messages = example['chosen']
        rejected_messages = example['rejected']

        # get the length of prompts
        prompt = example['chosen'][0]['content']
        prompt_template = tokenizer.apply_chat_template([{"content": prompt, "role": "user" }], tokenize=False, add_generation_prompt=True)
        # tokens_prompt = tokenizer.encode_plus(prompt_template, return_tensors="pt")['input_ids'][0]
        model_type = "llama3" if "llama" in  model_name else "gemma2"
        # print("model_type:", model_type)
        

        if check(chosen_messages, rejected_messages):
            # prompt = [m['role']+": "+m['content']+"\n" for m in chosen_messages[:-1]][0] 
            prompt = ""
            for m in chosen_messages[:-1]:
                prompt = prompt + m['role']+ ": " + m['content'] + "\n"

            response_chosen = chosen_messages[-1]["content"]
            response_rejected = rejected_messages[-1]["content"]
            # print(prompt)
            # print(response_chosen)
            # print(response_rejected)
        else:
            prompt, response_chosen, response_rejected = "", "", ""
            print("data", example['data_index'])

        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

        try:
            prompt_len = find_token_for_gating(tokens_chosen["input_ids"][0].tolist(), model_type)
        except:
            prompt_len = 1e9
        # print(tokenizer.convert_ids_to_tokens(tokens_chosen["input_ids"][0])[prompt_len-1:prompt_len+1])

        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0], "data_index": example['data_index'],
            "prompt": prompt, "chosen": response_chosen, "rejected": response_rejected,
            "prompt_plus_chosen_response": prompt_plus_chosen_response,
            "prompt_plus_rejected_response": prompt_plus_rejected_response,
            'prompt_length': prompt_len
        }

    ds = ds.map(formatting_func, batched=False, num_proc=8) 
    ds = ds.filter(lambda x: len(x["input_ids_chosen"]) <= script_args.max_length and len(x["input_ids_rejected"]) <= script_args.max_length, num_proc=8)
    
    len_before_filter = len(ds)
    ds = ds.filter(lambda x: x["prompt_length"] < script_args.max_length, num_proc=8)
    len_after_filter = len(ds)
    print(f"{len_after_filter-len_before_filter} prompts' lengths are over the prompt_reponse")
    
    remove_columns = []
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    
    return ds


# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
model_name_split = model_name.split("/")[-1]
output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}_{script_args.learning_rate}"

training_args = TrainingArguments(
    output_dir=os.path.join(output_name, 'logs'),
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    # weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=script_args.save_steps, #100
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=True, 
    remove_unused_columns=False,
    label_names=[],
    bf16=True,
    logging_strategy="steps",
    logging_steps=10,
    warmup_ratio=0.05,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    run_name=script_args.wandb_name,
    max_grad_norm=5.0,
    report_to='none',
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
)

# print("per_device_eval_batch_size:", training_args.per_device_eval_batch_size)

# Load the value-head model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = False)
tokenizer.model_max_length = script_args.max_length
# if 'gemma' not in model_name:
if 'Llama' in model_name or 'llama' in model_name:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
else:
    tokenizer.pad_token = tokenizer.eos_token

print("use build_dataset_mix")
dataset = build_dataset_mix(data_path, tokenizer, split='train') 
eval_dataset = dataset

#######################################################
print("Length of eval dataset:")
print(len(eval_dataset))

device = Accelerator().local_process_index 

print("device:", device)

if 'gemma' in model_name:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, device_map=device, 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
else:
    model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=1, device_map=device, 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

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
        prompt_length: Optional[torch.Tensor] = None
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print("model.device:", self.model.device)
        # print("input_ids:", input_ids.device)

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
        logits = self.score(hidden_states)

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
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths] # last token's logits

        loss = None
        
       
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
     
        chose_emb = hidden_states[0, sequence_lengths[0], :]
        rej_emb = hidden_states[1, sequence_lengths[1], :]
        prompt_emb = hidden_states[0, (prompt_length-1):(prompt_length+1), :] # last token of a prompt
        # print("check prompt:", (hidden_states[0, (prompt_length-1):(prompt_length+1), :]-hidden_states[1, (prompt_length-1):(prompt_length+1), :]).sum())
        # print("prompt_emb:", prompt_emb.shape)

        emb = torch.cat([chose_emb[None,...], rej_emb[None,...], prompt_emb], 0)
        
        # print("1.emb:", emb.shape, emb.sum())
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=emb,
            attentions=transformer_outputs.attentions,
        )

    
# hack model's forward
model.original_forward = model.forward
model.forward = custom_forward.__get__(model, type(model))


# Define the metric that we'll use for validation.
accuracy = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    predictions = np.argmax(predictions, axis=1)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    data_path: str = ""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        margins = []

        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            if 'margin' in feature.keys():
                margins.append(feature['margin'])
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if "preference_700K" in self.data_path or 'safe_pku' in self.data_path:
            batch = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "data_index": feature["data_index"],
                "prompt": feature["prompt"],
                "chosen": feature["chosen"],
                "rejected": feature["rejected"],
                "prompt_plus_chosen_response": feature["prompt_plus_chosen_response"],
                "prompt_plus_rejected_response": feature["prompt_plus_rejected_response"],
                "prompt_length": feature["prompt_length"]
            }
        else:
            batch = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "return_loss": True,
                "prompt": feature["prompt"],
                "chosen": feature["chosen"],
                "rejected": feature["rejected"],
                "source": feature["source"],
                "data_index": feature["data_index"],
                "prompt_plus_chosen_response": feature["prompt_plus_chosen_response"],
                "prompt_plus_rejected_response": feature["prompt_plus_rejected_response"],
                "prompt_length": feature["prompt_length"]
            }
        return batch

class RewardVisualizer(RewardTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        res = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], prompt_length=inputs["prompt_length"])
        # rewards, emb = res[0], res[2] # [2,1] ; [2, 803, 2048]
        rewards = res['logits']
        emb = res['hidden_states']
        # print("2.emb:", emb.shape, emb.sum())
        
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        ###################################
        if script_args.loss_type == 'origin':
            loss = - nn.functional.logsigmoid(rewards_j - rewards_k).mean() 
        elif script_args.loss_type == 'margin':
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k - torch.tensor(inputs["margin"], device=inputs["margin"][0].device).view(-1,1)).mean()
        elif script_args.loss_type == 'labelsmooth':
            loss = - 0.9 * nn.functional.logsigmoid(rewards_j - rewards_k).mean() - 0.1 * nn.functional.logsigmoid(rewards_k - rewards_j).mean() 
        else:
            raise NotImplementedError

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}, emb
        return loss, emb
    
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict, emb = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)

        # Stack accepted against rejected, 
        logits = torch.stack(logits).mean(dim=2).T 
        labels = torch.zeros(logits.shape[0])
        labels = self._prepare_inputs(labels)

        return loss, logits, labels, emb

    def visualize_samples(self, num_print_samples: int, cls_embs_path: str, data_path: str):
        """
        Visualize the reward model logits prediction

        Args:
            num_print_samples (`int`, defaults to `4`):
                The number of samples to print. Set to `-1` to print all samples.
        """
        eval_dataloader = self.get_eval_dataloader()
        table = defaultdict(list)
        # cls_embs_names = [] 
        if not os.path.exists(cls_embs_path):
            os.makedirs(cls_embs_path)

        print("Length of eval dataset:", len(self.eval_dataset))
        print("Length of eval dataloader:", len(eval_dataloader))
       
        for idx, inputs in tqdm.tqdm(enumerate(eval_dataloader)):
            data_index = inputs["data_index"]
            fn = os.path.join(cls_embs_path, f"emb_{data_index}.npy")
            if os.path.exists(fn):
                continue
            
            # print("inputs:", inputs['input_ids'].shape)
            _, logits, _, emb = self.prediction_step(self.model, inputs, prediction_loss_only=False)
            
            if 'preference_700K' in data_path or 'safe_pku' in data_path:
                source = " "
                chosen_text = inputs["chosen"]
                rejected_text = inputs["rejected"]
                prompt = inputs["prompt"]
            else:
                chosen_text = inputs["chosen"]
                rejected_text = inputs["rejected"]
                prompt = inputs["prompt"]
                source = inputs["source"]
            
            # print(Accelerator().local_process_index, data_index)

            table["prompt"].append(prompt)
            table["chosen_text"].append(chosen_text)
            table["rejected_text"].append(rejected_text)
            table["prompt_plus_chosen_response"].append(chosen_text)
            table["prompt_plus_rejected_response"].append(rejected_text)
            table["source"].append(source)
            table["data_index"].append(data_index.item())

            if num_print_samples >= 0 and len(table["chosen_text"]) >= num_print_samples:
                break
            
            if logits[0][0] > logits[0][1]:
                table["flag"].extend([1])
            else:
                table["flag"].extend([0])

            # save emb (2, num_token, c)
            cls_emb = emb.float().cpu().numpy() # (2, 2048)
            cls_emb = cls_emb[None,...]
            np.save(fn, cls_emb)
            # cls_embs_names.append(fn)
            if Accelerator().num_processes == 1:
                table["cls_emb"].extend(gather_object([fn]))
                table["logits"].extend(
                gather_object([[round(inner_item, 4) for inner_item in item] for item in logits.tolist()])
            )
            else:
                table["cls_emb"].append(fn)
                table["logits"].extend(
                [round(inner_item, 4) for inner_item in item] for item in logits.tolist()
            )

            # if Accelerator().process_index == 1:
            #     print(table)
    
            if len(table['chosen_text']) % 2e3 == 1:
                df = pd.DataFrame(table)
                df.to_csv(f"data_{os.path.basename(cls_embs_path)}_{Accelerator().local_process_index}.csv")

        df = pd.DataFrame(table)
        df.to_csv(f"data_{os.path.basename(cls_embs_path)}_{Accelerator().local_process_index}.csv")

        return df



if __name__ == "__main__":
    trainer = RewardVisualizer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=eval_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length, data_path=data_path),
    )

    cls_embs_path=script_args.cls_embs_path
    df = trainer.visualize_samples(1e8, cls_embs_path, data_path)
