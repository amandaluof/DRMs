from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from accelerate import Accelerator
# import evaluate
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    LlamaTokenizer,
    DataCollatorWithPadding,
)
from transformers.cache_utils import Cache
from transformers.utils import PaddingStrategy
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
# from utils import process_oass_dataset
os.environ["HF_TOKEN"] = 'xxxxxxxxxxx'

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_eval_batch_size: Optional[int] = field(default=8)
    max_length: Optional[int] = field(default=1024) #512) 
    base_model: Optional[str] = field(default="")
    rm_head: Optional[str] = field(default="")
    peft_name: Optional[str] =  field(default="")
    log_dir: Optional[str] = field(default='./eval_unified_reward_models_ood')
    task: Optional[str] = field(default='lmsys/mt_bench_human_judgments')
    freeze_pretrained: Optional[bool] = field(default=False)
    debug: Optional[bool] = field(default=False)
    normalize: Optional[bool] = field(default=False)
    normalize_statistics: Optional[str] = field(default="")
    cls_embs_path: Optional[str] = field(default='')

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model_name = script_args.base_model
tokenizer_name = model_name
data_path = script_args.task 

accelerator = Accelerator()
print(data_path)


def build_dataset(data_path, tokenizer, split='test', size=None):
    if 'rpr_multi_class_test' in data_path: 
        ds_tmp = None
        num_task = 5
        for i, attr in enumerate(['User-Friendliness', 'Narrative and Storytelling Quality', 'Linguistic Creativity', 'Scientific Rigor', 'Humor and Entertainment Value']):
            data_path = f"/srv/local/ry21/own_data/rpr_multi_class_test_with_hhh_format_{attr}.json"
            ds = load_dataset('json', data_files=data_path)['train']

            new_ds = ds.add_column('task_id', [i] * len(ds))
            if ds_tmp is None:
                ds_tmp = new_ds
            else:
                ds_tmp = concatenate_datasets([ds_tmp, new_ds])
        ds = ds_tmp
    else:
        ds = load_dataset(data_path, split=split)

    new_column = list(range(len(ds)))
    ds = ds.add_column("data_idx", new_column)

    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"padding": 'max_length', "truncation": True, "max_length": script_args.max_length, "return_tensors": "pt"}

        if 'HuggingFaceH4/hhh_alignment' in data_path or 'helpSteer2' in data_path or 'skywork_trainset' in data_path or 'rewardBench' in data_path or 'rpr' in data_path:
            if '\n\nAssistant:' in example['input'] or '\n\nHuman:' in example['input']:
                chosen_messages = []
                rejected_messages = []
                string_list = example['input'].split('\n\nHuman:')
                for lis in string_list:
                    if '\n\nAssistant:' in lis:
                        assistant_msg = lis.split('\n\nAssistant:')[1].strip()
                        human_msg = lis.split('\n\nAssistant:')[0].strip()
                        chosen_messages.extend([
                            {'role': 'user', 'content': human_msg},
                            {'role': 'assistant', 'content': assistant_msg},
                        ])
                        rejected_messages.extend([
                            {'role': 'user', 'content': human_msg},
                            {'role': 'assistant', 'content': assistant_msg},
                        ])
                    else: # last 
                        human_msg = lis.strip()
                        chosen_messages.extend([
                            {'role': 'user', 'content': human_msg},
                            {'role': 'assistant', 'content': example['targets']['choices'][np.argmax(example['targets']['labels'])]},
                        ])
                        rejected_messages.extend([
                            {'role': 'user', 'content': human_msg},
                            {'role': 'assistant', 'content': example['targets']['choices'][np.argmin(example['targets']['labels'])]},
                        ])
                prompt = ""
                for m in chosen_messages[:-1]:
                    prompt = prompt + m['role']+ ": " + m['content'] + "\n"
                response_chosen = chosen_messages[-1]["content"]
                response_rejected = rejected_messages[-1]["content"]
            else:
                chosen_messages = [
                    {'role': 'user', 'content': example['input']},
                    {'role': 'assistant', 'content': example['targets']['choices'][np.argmax(example['targets']['labels'])]},
                    ]
                rejected_messages = [
                    {'role': 'user', 'content': example['input']},
                    {'role': 'assistant', 'content': example['targets']['choices'][np.argmin(example['targets']['labels'])]},
                    ]
                prompt = "user: " + example['input']
                response_chosen = example['targets']['choices'][np.argmax(example['targets']['labels'])]
                response_rejected = example['targets']['choices'][np.argmin(example['targets']['labels'])]
        elif 'lmsys/mt_bench_human_judgments' in data_path:
            if example['winner'] == 'model_a':
                chosen_messages = example['conversation_a']
                rejected_messages = example['conversation_b']
            else:
                chosen_messages = example['conversation_b']
                rejected_messages = example['conversation_a']
        elif 'oass' in data_path:
            chosen_messages = example['chosen']
            rejected_messages = example['rejected']
        else:
            chosen_messages = [
                {'role': 'user', 'content': example['prompt']},
                {'role': 'assistant', 'content': example['response_{}'.format(example['better_response_id'])]},
                ]
            rejected_messages = [
                {'role': 'user', 'content': example['prompt']},
                {'role': 'assistant', 'content': example['response_{}'.format(1 - example['better_response_id'])]},
                ]

        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

        return {
            "input_ids": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],"prompt": prompt, "chosen": response_chosen, "rejected": response_rejected,
            "prompt_plus_chosen_response": prompt_plus_chosen_response,
            "prompt_plus_rejected_response": prompt_plus_rejected_response,
            "data_idx": example["data_idx"]
        }
    

    ds = ds.map(formatting_func, batched=False, num_proc=16)
    remove_columns = []
    for name in ds.column_names:
        if 'input_ids' not in name and 'attention' not in name and 'task_id' not in name and 'data_idx' not in name:
            remove_columns.append(name)

    prompts, responses_chosen, responses_rejected = ds['prompt'], ds['chosen'], ds['rejected']
    ds = ds.remove_columns(remove_columns)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= script_args.max_length and len(x["input_ids_rejected"]) <= script_args.max_length, num_proc=16)
    ds.set_format(type="torch")

    return ds, prompts, responses_chosen, responses_rejected


# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
model_name_split = model_name.split("/")[-1]

# Load the value-head model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = False)
tokenizer.model_max_length = script_args.max_length

###########################################################
if 'Llama' in model_name or 'llama' in model_name:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
else:
    tokenizer.pad_token = tokenizer.eos_token

eval_dataset, prompts, responses_chosen, responses_rejected = build_dataset(data_path, tokenizer, split='test')
if script_args.debug:
    eval_dataset = eval_dataset.select(range(0, 20))
print('size of test dataset: ', len(eval_dataset))
device = Accelerator().local_process_index 
print(device)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=1, device_map=device, 
    # load_in_8bit=True, 
    torch_dtype=torch.float16,
    # attn_implementation="flash_attention_2",
)

    
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

if script_args.rm_head:
    info = model.score.load_state_dict(torch.load(script_args.rm_head))
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
                
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        emb = hidden_states[0, sequence_lengths, :]
        # rej_emb = hidden_states[1, sequence_lengths[1], :]
        # emb = torch.cat([chose_emb[None,...], rej_emb[None,...]], 0)
        
        # print("1.emb:", emb.shape, emb.sum())
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=(emb, hidden_states),
            attentions=transformer_outputs.attentions,
        )

# hack model's forward
model.original_forward = model.forward
model.forward = custom_forward.__get__(model, type(model))


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
eval_data_loader = DataLoader(eval_dataset, batch_size=script_args.per_device_eval_batch_size, drop_last=False, collate_fn=data_collator)
eval_data_loader = accelerator.prepare(eval_data_loader)

full_prompts = []
full_chosen_responses = []
full_rejected_responses = []
full_chosen_prompts = []
full_rejected_prompts = []
full_rewards_chosen = []
full_rewards_rejected = []
full_task_ids = []
full_weights = []
full_npy_fns = []
pbar = tqdm(total=len(eval_dataset) // script_args.per_device_eval_batch_size // accelerator.num_processes)

if not os.path.exists(script_args.cls_embs_path):
    os.makedirs(script_args.cls_embs_path)

with torch.no_grad():
    for i, batch in enumerate(eval_data_loader):
        data_idx = batch['data_idx'].item()
        res = model(batch["input_ids"].to(model.device), attention_mask=batch["attention_mask_chosen"].to(model.device))
        chosen_emb, chose_hidden_states = res["hidden_states"]
        res = model(batch["input_ids_rejected"].to(model.device), attention_mask=batch["attention_mask_rejected"].to(model.device))
        rej_emb, rej_hidden_states = res["hidden_states"]

        # embedding
        emb = torch.cat([chosen_emb, rej_emb], 0)
        cls_emb = emb.float().cpu().numpy() # (2, 2048)
        cls_emb = cls_emb[None,...]
        fn = os.path.join(script_args.cls_embs_path, f"emb_{data_idx}.npy")
        np.save(fn, cls_emb)

        # hidden states
        hidden_states = torch.cat([chose_hidden_states, rej_hidden_states], 0)
        hidden_states = hidden_states.float().cpu().numpy()
        fn2 = os.path.join(script_args.cls_embs_path, f"hs_{data_idx}.npy")
        np.save(fn2, hidden_states)

        input_ids = torch.cat([batch["input_ids"], batch["input_ids_rejected"]])
        input_ids = input_ids.int().cpu().numpy()
        fn3 = os.path.join(script_args.cls_embs_path, f"ii_{data_idx}.npy")
        np.save(fn3, input_ids)

        prompt = prompts[data_idx]
        response_chosen = responses_chosen[data_idx]
        response_rejected = responses_rejected[data_idx]

        full_prompts.extend([prompt])
        full_chosen_responses.extend([response_chosen])
        full_rejected_responses.extend([response_rejected])
        full_chosen_prompts.extend(batch['input_ids'])
        full_rejected_prompts.extend(batch['input_ids_rejected'])
        if 'task_id' in batch.keys():
            full_task_ids.extend(batch['task_id'])
        full_npy_fns.extend([fn])
        pbar.update(1)


full_chosen_prompts = tokenizer.batch_decode(full_chosen_prompts)
full_chosen_prompts = [x.rstrip('[PAD]') for x in full_chosen_prompts]
full_rejected_prompts = tokenizer.batch_decode(full_rejected_prompts)
full_rejected_prompts = [x.rstrip('[PAD]') for x in full_rejected_prompts]
full_rewards_chosen = [x.item() for x in full_rewards_chosen]
full_rewards_rejected = [x.item() for x in full_rewards_rejected]
if len(full_task_ids):
    full_task_ids = [x.item() for x in full_task_ids]

if len(full_weights) > 0:
    full_weights = torch.cat(full_weights, 0) # (N, 12)

# print(full_chosen_prompts)

accelerator.wait_for_everyone()
all_full_prompts = accelerator.gather_for_metrics(full_prompts)
all_full_chosen_responses = accelerator.gather_for_metrics(full_chosen_responses)
all_full_rejected_responses = accelerator.gather_for_metrics(full_rejected_responses)
all_chosen_prompts = accelerator.gather_for_metrics(full_chosen_prompts)
all_rejected_prompts = accelerator.gather_for_metrics(full_rejected_prompts)
all_full_npy_fns = accelerator.gather_for_metrics(full_npy_fns)
if len(full_task_ids):
    all_task_ids = accelerator.gather_for_metrics(full_task_ids)
if len(full_weights) > 0:
    all_full_weights =  accelerator.gather_for_metrics(full_weights)


if accelerator.is_main_process:
    evaluation_result = {
        'prompts': all_full_prompts,
        'chosen_responses': all_full_chosen_responses,
        'rejected_responses': all_full_rejected_responses,
        'cls_emb': all_full_npy_fns
    }
    if len(full_task_ids):
        evaluation_result['task_ids'] = all_task_ids
    dataframe = pd.DataFrame(evaluation_result)
    dataframe.to_csv(os.path.join(f'data_sample_{model_name.split("/")[1]}_{script_args.task.split("/")[-1]}.csv'))    


