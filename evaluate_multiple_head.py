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

'''
Calculate all heads' scores and save them in json.
For rewardBench, save each head's result in a json belonging to the same dir
For hhh, save all head's result in a single json 
'''

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
from safetensors import safe_open
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import Any, Dict, List, Optional, Union, Tuple
import transformers
from transformers.cache_utils import Cache
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
import pandas as pd
import json

import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("./reward-bench")

from rewardbench import (
    REWARD_MODEL_CONFIG,
    torch_dtype_mapping,
    save_to_hub,
)
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import AutoModelForSequenceClassification
from score_head import MutipleHead
from safetensors.torch import load_file

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
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
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
        "--score_head_type", type=str, help="type of score head"
    )
    parser.add_argument(
        "--score_head_weight", type=str, help="path to moe model"
    )
    parser.add_argument(
        "--normalize", action="store_true", help="normalize scores"
    )
    parser.add_argument(
        "--normalize_statistics", type=str, help="path to moe score statictics"
    )
    parser.add_argument(
        "--emb_path", type=str, help="path to emb get from original reward model, rewardBench use this param"
    )
    parser.add_argument(
        "--res_csv", type=str, help="path to emb get from original reward model, hhh and helpSteer use this param"
    )
    parser.add_argument(
        "--task", type=str, help="rewardBench or rpr"
    )
    parser.add_argument("--num_head", type=int, default=64, help="number of reward heads")
    parser.add_argument("--seed", type=int, default=42, help="seed for random initialize")
    parser.add_argument("--visualize", action="store_true", help="whether this is a visualization process or not")

    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
    return args


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

    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    if args.model in REWARD_MODEL_CONFIG:
        config = REWARD_MODEL_CONFIG[args.model]
    else:
        config = REWARD_MODEL_CONFIG["default"]
    # logger.info(f"Using reward model config: {config}")

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


    torch_dtype = config.get("torch_dtype", None)
    # if not datatype in config (default), check args
    if torch_dtype is None:
        # if datatype is bfloat16, then manually turn off quantizaiton (done with bitsandbytes)
        if args.torch_dtype == torch.bfloat16:
            quantized = False
            logger.info("Disabling quantization for bfloat16 datatype")
        torch_dtype = args.torch_dtype

    ############################
    # Load dataset
    ############################
    if args.task == "rewardBench":
        emb_path = args.emb_path
        with open(emb_path, "rb") as f:
            dataset = pickle.load(f)

        ids = dataset["ids"]
        subsets = dataset["subsets"]
        embs = dataset["embs"]

    elif args.task == "rpr_multi_class_test":
        dataframe = pd.read_csv(args.res_csv)
        cls_emb_paths = dataframe["cls_emb"]
        embs = []
        for cls_emb_path in tqdm(cls_emb_paths):
            try:
                emb = np.load(cls_emb_path)
            except:
                # print("reload data from /srv/local")
                emb = np.load(cls_emb_path.replace("/home", "/srv/local"))
            if len(emb.shape) == 2:
                emb = emb[None, ...]
            embs.append(emb) # (1, 2, 2048)
        embs = np.concatenate(embs, 0)
        embs = torch.Tensor(embs)
    embs = embs.to(accelerator.process_index, dtype=torch.float16)

    ############################
    # Load model
    ############################
    print("Load model:", args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=1, #device_map=device, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            )
    hidden_size = model.config.hidden_size
    
    ############################
    # Load head
    ############################
    model_type=args.score_head_type
    num_head = args.num_head

    if model_type == "DRM":
        model.score = MutipleHead(hidden_size=hidden_size, output_size=num_head, score_head_weight=args.score_head_weight, dtype=torch.bfloat16, device=accelerator.process_index, model_type=model_type)
        print("Modified model.score to use MutipleHead.") 
        print("Use DRM")
    
    model.to(accelerator.process_index, dtype=torch.float16)
    model.eval()
    logger.info("*** Load reward model ***")

    model = accelerator.prepare(model)
    
    rewards = model.score(embs)

    if isinstance(rewards, tuple):
        rewards_chosen, rewards_rejected = rewards
        rewards = torch.cat([rewards_chosen[:,None], rewards_rejected[:,None]], 1)

    all_rewards_chosen = rewards[:,0].float().cpu().detach().numpy()
    all_rewards_rejected = rewards[:,1].float().cpu().detach().numpy()

    if args.task == "rewardBench":        
        for i in range(num_head):
            rewards_chosen = all_rewards_chosen[:, i].tolist()
            rewards_rejected = all_rewards_rejected[:, i].tolist()
            results = []
            [
                results.append(1) if chosen > rejected else results.append(0)
                for chosen, rejected in zip(rewards_chosen, rewards_rejected)
            ]

            # # get core dataset
            results_grouped = {}
                    
            # print per subset and log into results_grouped file
            present_subsets = np.unique(subsets)

            for subset in present_subsets:
                # subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
                subset_results = [r for r, s in zip(results, subsets) if s == subset]
                num_correct = sum(subset_results)
                num_total = len(subset_results)
                results_grouped[subset] = num_correct / num_total

            # log leaderboard aggregated results
            if not args.pref_sets:
                results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
                results_leaderboard["overall"] = sum(results_leaderboard.values()) / len(results_leaderboard)
                print(args.score_head_weight)
                print(results_leaderboard)

            sub_path = "eval-set/" if not args.pref_sets else "pref-sets/"
            
        
            case_name = args.model.split("/")[-1] + args.score_head_type + f"head_{num_head}"

            save_json_dir = case_name
            os.makedirs(save_json_dir, exist_ok=True)
            save_json_name = os.path.join(save_json_dir, case_name+f"_{i}.json") # total score
            
            scores_dir = f"/srv/local/ry21/own_code/DRMs/reward-bench/results/eval-set-scores/{case_name}"

            os.makedirs(scores_dir, exist_ok=True)
            scores_path = os.path.join(scores_dir, case_name+f"_{i}.json")

            def save_res(
                results_dict: Union[Dict, List],
                json_path: str,
                target_path: str,
                save_metrics_for_beaker: bool = False,
            ):
                scores_path = os.path.join("results/", f"{target_path}{json_path}") # /home/fl38/own_code/reward_train_huan/reward-bench/results/

                if save_metrics_for_beaker:
                    # ai2 internal visualization, not needed externally, global path intentional.
                    dirname = os.path.dirname("output/metrics.json")
                    os.makedirs(dirname, exist_ok=True)  # redundant in Beaker code
                    with open("output/metrics.json", "w+") as f:  # save format for AI2 beaker to show results
                        json.dump(results_dict, f)

                dirname = os.path.dirname(scores_path)
                os.makedirs(dirname, exist_ok=True)

                # remove old data
                if os.path.isfile(scores_path):
                    os.remove(scores_path)

                with open(scores_path, "w") as f:
                    if isinstance(results_dict, Dict):
                        dumped = json.dumps(results_dict, indent=4, sort_keys=True)  # nol removed , default=str
                        f.write(dumped)
                    # else, dump each row in list
                    else:
                        for record in results_dict:
                            dumped = json.dumps(record, indent=4, sort_keys=True) + "\n"
                            f.write(dumped)

                return None

            results_url = save_res(
                results_grouped,
                save_json_name,
                sub_path,
                save_metrics_for_beaker=True,
            )   
            # this is a template
            scores_dict = json.load(open("/srv/local/ry21/own_code/reward_train_huan/reward-bench/results/eval-set-scores/home/ry21/own_code/reward_train/reward_models_sft/Gemma-2B-rewardmodel-baseline-emb-mixture2_and_safe_pku-PCA-component/Gemma-2B-rewardmodel-baseline-emb-mixture2_and_safe_pku-PCA-component0.json", 'r'))
            scores_dict['scores_chosen'] = [[s] for s in rewards_chosen]
            scores_dict['scores_rejected'] = [[s] for s in rewards_rejected]

            with open(scores_path, "w") as f:
                dumped = json.dumps(scores_dict, indent=4, sort_keys=True)  # nol removed , default=str
                f.write(dumped)
            print(f"save result to {scores_path}")
        print(f"save scores of all heads to {scores_dir}")

    elif args.task == "rpr_multi_class_test":
        results_hhh = {}
        for head in range(num_head):
            head_scores_chosen = all_rewards_chosen[:, head]
            head_scores_rejected = all_rewards_rejected[:, head]

            overall_accuracy = round(float((head_scores_chosen > head_scores_rejected).mean()), 3)

            if args.task == "rpr_multi_class_test":
                num_task = 5
                
            per_task_accuracy = {}
            for t in range(num_task): # task_id: [0, 1, 2, 3]
                task_mask = dataframe['task_ids']==t
                task_accuracy = float((head_scores_chosen[task_mask] > head_scores_rejected[task_mask]).mean())
                task_accuracy = round(task_accuracy, 3)
                per_task_accuracy[str(t)] = task_accuracy
            
            results_hhh[f"head_{head}"] = {
                "scores_chosen": head_scores_chosen.tolist(),
                "scores_rejected": head_scores_rejected.tolist(),
                "overall_accuracy": overall_accuracy,
                "per_task_accuracy": per_task_accuracy,
                "task_ids": dataframe['task_ids'].tolist()
            }
            if args.task == "rpr_multi_class":
                print(f"[rpr_multi_class] Head {head}: overall accuracy: {overall_accuracy}, per task: {per_task_accuracy}")
        
        if args.task == "rpr_multi_class_test":
            save_dir = os.path.join("..", "rpr_multi_class_alignment", args.model.split("/")[-1], model_type) 
            os.makedirs(save_dir, exist_ok=True)
            print(save_dir)           
            json_save_path = os.path.join(save_dir, f"rpr_multi_class_head_scores_num_head_{num_head}.json")
            with open(json_save_path, "w") as f:
                json.dump(results_hhh, f, indent=4, sort_keys=True)
            print(f"[rpr_multi_class] save head scores to: {json_save_path}")
    else:
        NotImplementedError




if __name__ == "__main__":
    main()
