import os
import json
import glob
import pickle
import argparse

import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section

def evaluate():
    return


def list_to_scalar(x):
    if isinstance(x, list) and len(x) == 1:
        return x[0]
    return x


def get_values(x):
    return x[0]


def calculate_statistics(data):
    """
    Calculate the mean and variance for each key in a list of dictionaries.

    Parameters:
    data (list of dict): A list of dictionaries with numerical values.

    Returns:
    tuple: A tuple containing two dictionaries:
           - 'averages': Dictionary with the mean of each key.
           - 'variances': Dictionary with the variance of each key.
    """
    import numpy as np
    
    # Ensure there is data to process
    if not data:
        return {"averages": {}, "standards": {}}
    
    # Extract keys from the first dictionary
    keys = data[0].keys()
    averages = {}
    standards = {}

    # Calculate mean and variance for each key
    for key in keys:
        values = [item[key] for item in data]
        averages[key] = round(np.mean(values).item(), 3)
        standards[key] = round(np.std(values).item(), 4)

    return {"averages": averages, "standards": standards}


def standardize_matrix_torch(matrix, num_sample):
    """
    使用PyTorch对矩阵在行维度进行标准化 (z-score scaling)
    :param matrix: 输入矩阵，形状为 (N, M), (115, 22)  (num_category*k, 2*num_head)
    :param num_sample: the number of sample we have for each data subset
    :return: 标准化后的矩阵
    """
    num_model = int(matrix.shape[-1] / 2)
    num_subset = int(matrix.shape[0] / num_sample)
    matrix_group = matrix.reshape((num_subset, num_sample, 2, num_model)) # (23, 5, 2, 11)
    matrix_group = matrix_group.reshape(num_subset, -1, num_model) # (23, 10, 11)
    mean_vals = torch.mean(matrix_group, dim=1)  # 每个模型在每个subset的分数均值, (num_subset, num_model)
    std_vals = torch.std(matrix_group, dim=1)   # 每个模型在每个subset的分数标准差, (num_subset, num_model)
    # 防止标准差为0
    std_vals[std_vals == 0] = 1

    matrix = matrix.reshape(num_subset, num_sample, 2, num_model).permute((1,2,0,3)).reshape((num_sample*2, num_subset, num_model))
    standardized_matrix = (matrix - mean_vals[None,...]) / std_vals[None,...] 
    standardized_matrix = standardized_matrix.reshape((num_sample, 2, num_subset, num_model)).permute((2, 0, 1, 3))
    standardized_matrix = standardized_matrix.reshape((num_subset*num_sample, 2*num_model))    

    return standardized_matrix, mean_vals, std_vals


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, help="rewardBench or hhh"
    )
    parser.add_argument(
        "--result_path", type=str, help="rewardBench or hhh"
    )
    parser.add_argument(
        "--score_head_type", type=str, help="rewardBench or hhh"
    )
    parser.add_argument(
        "--num_eval_head", type=int, nargs='+', help="number of head"
    )
    parser.add_argument(
        "--original_head_result_path", type=str, help="original_head_result_path"
    )
    parser.add_argument(
        "--category_based", action="store_true", help="calculate weight for each category instead of each set"
    )
    parser.add_argument(
        "--wo_original_head", action="store_true", help="whether the original head is needed"
    )
    parser.add_argument(
        "--res_name", type=str, help="result name of the metrics"
    )
    parser.add_argument(
        "--score_delta", action="store_true"
    )
    args = parser.parse_args()
    return args

def custom_sample(group, k, seed):
    n = len(group)
    if n < k:
        return group.sample(n=k, replace=True, random_state=seed)
    else:
        return group.sample(n=k, replace=False, random_state=seed)


if __name__ == "__main__":
    temperatures = [1] 
    metrics = dict()
    info = ""
    args = get_args()

    if args.task == "rpr_multi_class_test":
        ks = [5] 
    else:
        ks = [15] 

    if args.num_eval_head:
        num_eval_heads = args.num_eval_head
    
    print("ks:", ks)
    print("num_eval_heads:", num_eval_heads)

    for num_eval_head in num_eval_heads:
        ###### Load dataset and rename columns ######
        if args.task == "rewardBench":
            res_PCA_folder = args.result_path
            json_files = glob.glob(os.path.join(res_PCA_folder, "*.json"))
            json_files = sorted(json_files, key=lambda x:int(x.split("_")[-1].split(".")[0]))
            num_total_head = len(json_files)
            
            if args.score_head_type == "DRM":
                pos_json_files = json_files[:int(len(json_files)/2)]
                neg_json_files = json_files[int(len(json_files)/2):]
                if args.wo_original_head:                
                    json_files = pos_json_files[:int(num_eval_head/2)] + neg_json_files[:int(num_eval_head/2)]
                else:
                    if args.original_head_result_path is not None:
                        json_files = pos_json_files[:int(num_eval_head/2)] + neg_json_files[:int(num_eval_head/2)-1]
                        json_files.insert(0, args.original_head_result_path)
                    else:
                        json_files = pos_json_files[:int(num_eval_head/2)] + neg_json_files[:int(num_eval_head/2)]
            else:
                json_files = json_files[:num_eval_head]
            # print(json_files)
            print(f"Use {len(json_files)} files to calculate TTA results.")

            num_head = len(json_files)
            df_merged = None


            for i, file_path in enumerate(json_files):
                with open(file_path, "r") as f:
                    data = json.load(f)  # data 假设是一个 dict，比如 {"id": [...], "scores_chosen": [...], "scores_rejected": [...]} 
                    if "results_leaderboard" in data.keys():
                        data.pop("results_leaderboard")
                    df_tmp = pd.DataFrame(data)

                    if not args.wo_original_head and args.score_delta and i==0:
                        df_tmp['scores_delta'] = df_tmp['scores_chosen'] - df_tmp['scores_rejected']


                    if args.score_delta:
                        df_tmp = df_tmp.rename(columns={
                            "scores_chosen": f"scores_chosen_{i}",
                            "scores_rejected": f"scores_rejected_{i}",
                            "scores_delta": f"scores_delta_{i}"
                        })
                    else:
                        df_tmp = df_tmp.rename(columns={
                            "scores_chosen": f"scores_chosen_{i}",
                            "scores_rejected": f"scores_rejected_{i}"
                        })

                    if df_merged is None:
                        df_merged = df_tmp
                    else:
                        # 按 id 列来 merge，如果没有 id 列，需要你根据实际数据做调整
                        if args.score_delta:
                            df_merged = pd.concat([df_merged, 
                            df_tmp[[f"scores_chosen_{i}", f"scores_rejected_{i}", f"scores_delta_{i}"]]], axis=1)
                        else:
                            df_merged = pd.concat([df_merged, 
                            df_tmp[[f"scores_chosen_{i}", f"scores_rejected_{i}"]]], axis=1)

            scores_columns = []
            for i in range(num_eval_head):
                if args.score_delta:
                    scores_columns.extend([f"scores_chosen_{i}", f"scores_rejected_{i}", f"scores_delta_{i}"])
                else:
                    scores_columns.extend([f"scores_chosen_{i}", f"scores_rejected_{i}"])
            for scores_column in scores_columns:
                df_merged[scores_column] = df_merged[scores_column].apply(list_to_scalar)
                

        elif args.task == "rpr_multi_class_test":
            combined_results_path = args.result_path
            combined_results = json.load(open(combined_results_path, 'r'))
            num_total_head = len(combined_results)
            print("length of heads in the original file:", num_total_head)

            if args.score_head_type == "DRM":
                # extract part of combined_results based on head, [0, 4095] with [0, 2047] for pos head and [2048, 4095] for neg head
                extract_head_names1 = [f"head_{i}" for i in range(int(num_eval_head/2))]
                if getattr(args, "wo_original_head", False):
                    # 当不考虑原始head时，直接用固定切片
                    extract_head_names2 = [f"head_{i}" for i in range(num_total_head)[
                        int(num_total_head/2) : int(num_total_head/2) + int(num_eval_head/2)
                    ]]
                else:
                    if num_eval_head%2 == 0:
                        extract_head_names2 = [f"head_{i}" for i in range(num_total_head)[int(num_total_head/2):int
                        (num_total_head/2)+int(num_eval_head/2)]][:-1] # 
                    else:
                        extract_head_names2 = [f"head_{i}" for i in range(num_total_head)[int(num_total_head/2):int
                        (num_total_head/2)+int(num_eval_head/2)]]
                extract_head_names = extract_head_names1 + extract_head_names2
                combined_results = {key: combined_results[key] for key in extract_head_names if key in combined_results}
                # print(len(combined_results.keys()))
                
                if hasattr(args, "original_head_result_path") and args.original_head_result_path:
                    ori_head_result_path = args.original_head_result_path
                    ori_head_results = json.load(open(ori_head_result_path, 'r'))
                    if args.score_delta:
                        ori_head_results['head_0']['scores_delta'] = [ori_head_results['head_0']['scores_chosen'][i] - ori_head_results['head_0']['scores_rejected'][i] for i in range(len(ori_head_results['head_0']['scores_chosen']))]

                    existing_heads = set(int(key.split('_')[1]) for key in ori_head_results.keys())
                    new_head_start = max(existing_heads) + 1 if existing_heads else 0
                    new_combined_results = ori_head_results.copy()
                else:
                    new_head_start = 0
                    new_combined_results = {}

                for head_key, head_data in combined_results.items():
                    head_num = int(head_key.split('_')[-1])  # 获取原 head 数字
                    new_key = f"head_{new_head_start}"  # 重新编号
                    new_head_start += 1  # 递增
                    new_combined_results[new_key] = head_data
                combined_results = new_combined_results

            merge_data = dict()
            for head_key, head_data in combined_results.items():
                head_num = head_key.split('_')[-1]
                scores_chosen = head_data.pop("scores_chosen", None)
                scores_rejected = head_data.pop("scores_rejected", None)
                scores_delta = head_data.pop("scores_delta", None)
                
                if scores_chosen is not None:
                    merge_data[f"scores_chosen_{head_num}"] = scores_chosen
                if scores_rejected is not None:
                    merge_data[f"scores_rejected_{head_num}"] = scores_rejected
                if scores_delta is not None:
                    merge_data[f"scores_delta_{head_num}"] = scores_delta


            merge_data['subset'] = head_data['task_ids']
            df_merged = pd.DataFrame(merge_data)
            num_head = len(combined_results)

            scores_columns = []
            for i in range(num_eval_head):
                if args.score_delta:
                    scores_columns.extend([f"scores_chosen_{i}", f"scores_rejected_{i}", f"scores_delta_{i}"])
                else:
                    scores_columns.extend([f"scores_chosen_{i}", f"scores_rejected_{i}"])

        scores_chose_columns = []
        for i in range(num_eval_head):
            scores_chose_columns.extend([f"scores_chosen_{i}"])
        scores_rejected_columns = []
        for i in range(num_eval_head):
            scores_rejected_columns.extend([f"scores_rejected_{i}"])
        
        if args.score_delta:
            scores_delta_columns = []
            for i in range(num_eval_head):
                scores_delta_columns.extend([f"scores_delta_{i}"])

        print(f"use {len(scores_chose_columns)}  to calculate TTA results.")
        print("score_delta:", args.score_delta)

        # apply reweight
        for k in ks: 
            for temperature in temperatures:
                seeds = range(20)
                # seeds = range(1)
                dict_list = []
                weight_list = []
                normalize = True

                for seed in seeds:
                    # calculate weight
                    if args.category_based:
                        reverse_mapping = {value: key for key, values in SUBSET_MAPPING.items() for value in values}
                        df_merged['category'] = df_merged['subset'].map(reverse_mapping)
                        df_sampled = (
                            df_merged
                            .groupby('category', group_keys=False) 
                            .apply(lambda g: custom_sample(g, k=k, seed=seed), include_groups=True)
                    )
                        group_names = df_sampled['category'].tolist()[::k]
                    else:
                        # subset based
                        df_sampled = (
                            df_merged
                            .groupby('subset', group_keys=False) 
                            .apply(lambda g: custom_sample(g, k=k, seed=seed), include_groups=True)
                    )
                        group_names = df_sampled['subset'].tolist()[::k]

                    score_chosen = torch.Tensor(np.array(df_sampled[scores_chose_columns]))
                    score_rejected = torch.Tensor(np.array(df_sampled[scores_rejected_columns]))

                    if args.score_delta:
                        score_delta = torch.Tensor(np.array(df_sampled[scores_delta_columns]))

                    # normalize
                    if normalize:
                        scores = torch.cat([score_chosen, score_rejected], -1) # ['scores_chosen_0', ..., 'scores_chosen_9', 'scores_rejected_0', ..., 'scores_rejected_9']
                        scores, mean_scores, std_scores = standardize_matrix_torch(scores, k) # scores (num_category*k, 2*num_head)
                        
                        # mean_scores = np.repeat(mean_scores.numpy(), 2, 1)
                        # std_scores = np.repeat(std_scores.numpy(), 2, 1)
                        mean_scores = mean_scores.numpy()
                        std_scores = std_scores.numpy()
                        
                        score_chosen, score_rejected = scores[:, :int(scores.shape[1]/2)], scores[:, int(scores.shape[1]/2):]

                    if args.score_delta:
                        print("use delta to calculate weight")
                        loss = - nn.functional.logsigmoid(score_delta) 
                    else:
                        loss = - nn.functional.logsigmoid(score_chosen - score_rejected) 
    
                    num_model = loss.shape[-1]
                    loss = loss.reshape((-1, k, num_model))
                    loss = loss.sum(1)


                    prior_weight = np.ones(loss.shape) / num_model
                    weight = (-loss/temperature).softmax(dim=-1).numpy()
                    weight_list.append(weight)

                    # Apply weight
                    scores = df_merged[scores_columns].copy() # [chose_0, reject_0, ...chose_9, reject_9,]
                    scores['agg_chose'] = scores['scores_chosen_0'] * 0.0
                    scores['agg_rejected'] = scores['scores_rejected_0'] * 0.0
                    if args.score_delta:
                        scores['agg_delta'] = scores['scores_delta_0'] * 0.0
                    
                    for idx, group_name in enumerate(group_names):
                        if args.category_based:
                            mask = (df_merged['category'] == group_name)
                        else:
                            mask = (df_merged['subset'] == group_name)
                        
                        if normalize:
                            chose_scores_head = scores[scores_chose_columns]
                            scores.loc[mask, 'agg_chose'] = ((chose_scores_head.loc[mask] - mean_scores[idx]) / std_scores[idx] * weight[idx]).values.sum(-1)
                            reject_scores_head = scores[scores_rejected_columns]
                            scores.loc[mask, 'agg_rejected'] = ((reject_scores_head.loc[mask] - mean_scores[idx]) / std_scores[idx] * weight[idx]).values.sum(-1)
                        else:
                            chose_scores_head = scores[scores_chose_columns]
                            scores.loc[mask, 'agg_chose'] = (chose_scores_head.loc[mask] * weight[idx]).values.sum(-1)
                            reject_scores_head = scores[scores_rejected_columns]
                            scores.loc[mask, 'agg_rejected'] = (reject_scores_head.loc[mask] * weight[idx]).values.sum(-1)
                        
                        if args.score_delta:
                            delta_scores_head = scores[scores_delta_columns]
                            scores.loc[mask, 'agg_delta'] = (delta_scores_head.loc[mask] * weight[idx]).values.sum(-1)

                    score_agg_chose =  scores['agg_chose']
                    score_agg_rejected = scores['agg_rejected']
                    if args.score_delta:
                        score_agg_delta =  scores['agg_delta']

                    # calculate metric
                    if args.task == "rewardBench":                    
                        # generate_json_file
                        results_grouped = dict()
                        results = []
                        if args.score_delta:
                            [
                                            results.append(1) if delat_score > 0 else results.append(0)
                                            for delat_score in score_agg_delta
                                        ]
                        else:
                            [
                                            results.append(1) if chosen > rejected else results.append(0)
                                            for chosen, rejected in zip(score_agg_chose, score_agg_rejected)
                                        ]
                        init_data = json.load(open(json_files[0], 'r'))
                        try:
                            init_data = pd.DataFrame(init_data)
                        except:
                            init_data.pop("results_leaderboard")
                            init_data = pd.DataFrame(init_data)
                        init_data['results'] = results
                        subsets = init_data['subset']
                        present_subsets = np.unique(subsets)
                        for subset in present_subsets:
                            subset_dataset = init_data[init_data["subset"] == subset]
                            num_correct = sum(subset_dataset["results"].to_list())
                            num_total = len(subset_dataset["results"])
                            # print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
                            results_grouped[subset] = num_correct / num_total

                        # log leaderboard aggregated results
                        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
                        results_leaderboard["overall"] = sum(results_leaderboard.values()) / len(results_leaderboard)
                        # print(f"N={k}", f"temperature={temperature}")
                        # print(results_leaderboard)

                        def formatting(results):
                            for k,v in results.items():
                                results[k] = round(v, 3)
                            return results

                        results_leaderboard = formatting(results_leaderboard)
                        dict_list.append(results_leaderboard)
                    elif args.task == "rpr_multi_class_test":
                        if args.score_delta:
                            overall_accuracy = round(float((score_agg_delta > 0).mean()), 3)    
                        else:
                            overall_accuracy = round(float((score_agg_chose > score_agg_rejected).mean()), 3)

                        per_task_accuracy = {}
                        if args.task == "rpr_multi_class_test":
                            num_task = 5
                            task_names = ['User-Friendliness', 'Narrative and Storytelling Quality', 'Linguistic Creativity', 'Scientific Rigor', 'Humor and Entertainment Value']

                        for t in range(num_task): # task_id: [0, 1, 2, 3]
                            task_mask = df_merged['subset']==t
                            if args.score_delta:
                                task_accuracy = float((score_agg_delta[task_mask] > 0).mean())
                            else:
                                task_accuracy = float((score_agg_chose[task_mask] > score_agg_rejected[task_mask]).mean())
                            task_accuracy = round(task_accuracy, 3)
                            per_task_accuracy[task_names[t]] = task_accuracy

                        
                        results_hhh = {
                            "scores_chosen": score_agg_chose.tolist(),
                            "scores_rejected": score_agg_rejected.tolist(),
                            "overall_accuracy": overall_accuracy,
                            "per_task_accuracy": per_task_accuracy,
                            "subset": df_merged['subset'].tolist()
                        }
                        # print(f"[hhh] Head : overall accuracy: {overall_accuracy}, per task: {per_task_accuracy}")
                        result = per_task_accuracy.copy()
                        result["overall"] = overall_accuracy
                        dict_list.append(result)


                result = calculate_statistics(dict_list)
                info = info + "\n" + f"Num_head={num_eval_head} N={k} " + f"temperature={temperature} mean" + str(result['averages'])
                info = info + "\n" + f"Num_head={num_eval_head} N={k} " + f"temperature={temperature} std" + str(result['standards'])
                # print(result['variances'])

                weights = np.array(weight_list)
                metrics[f"{num_eval_head}_{k}_{temperature}"] = (result, weights.tolist())
    
    try:
        print("res_PCA_folder:", res_PCA_folder)
    except:
        print("combined_results_path:", combined_results_path)
    print("score_head_type:", args.score_head_type, "num_head:", weights.shape[-1], "normalize:", normalize)
    print(info)

    if args.res_name is not None:
        with open(os.path.join("results", args.res_name), 'wb') as f:
            pickle.dump(metrics, f)
        print(f"save result to results/{args.res_name}")

    

