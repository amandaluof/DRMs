import os
import glob
import tqdm
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description="Choose dimensionality reduction method")
    parser.add_argument('--basemodel_names', type=str, nargs='+', help="List of base model names")
    parser.add_argument('--case_names', type=str, nargs='+', help="List of case names")
    parser.add_argument(
        "--decompose_method",
        type=str,
        choices=["PCA"],
    )
    parser.add_argument(
        "--num_component",
        type=int,
        default=20,
        help="The number of components to get when decomposed"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--full_composed",
        action="store_true",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parse()
    basemodel_names = args.basemodel_names 
    case_names = args.case_names

    root = "/srv/local/ry21/own_code/reward_models_sft"  
    npy_root = "/srv/local/ry21/own_data"

    for idx in range(len(case_names)):
        case_name = case_names[idx]
        model_name = basemodel_names[idx]

        npy_dir = f"{npy_root}/{case_name}"
        npy_filenames = glob.glob(os.path.join(npy_dir, "emb_*.npy"))
        npy_arrays = []
        
        merge_path = f"/srv/local/ry21/own_data/{case_name}"

        if not os.path.exists(merge_path):
            os.makedirs(merge_path)
        merge_array_name = os.path.join(merge_path, f"{case_name}.npy")

        if os.path.exists(merge_array_name):
            merged_array = np.load(merge_array_name)
            print(f"Loading from exsiting file {merge_array_name}")
            print("merged_array:", merged_array.shape)
        else:
            for filename in tqdm.tqdm(npy_filenames):
                array = np.load(filename)  
                npy_arrays.append(array)  
            merged_array = np.concatenate(npy_arrays, axis=0)
            np.save(merge_array_name, merged_array)
            print(f"save merged_array to {merge_array_name}")
            print("merged_array:", merged_array.shape)
        
        vectors = merged_array # (N,2,d)
        
        diff = vectors[:, 0, :] - vectors[:, 1, :]
        print("diff shape:", diff.shape)

        matrix = diff
        if args.full_composed:
            k = matrix.shape[-1]
        else:
            k = args.num_component

        pca = PCA(n_components=k) # svd_solver='covariance_eigh' 
        reduced_matrix = pca.fit_transform(matrix)
        explained_variance_ratio = pca.explained_variance_ratio_
        print(f"各主成分的方差解释比例: {explained_variance_ratio}")
        components_vector = pca.components_
        eigenvalues = pca.explained_variance_
    

        model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=1, #device_map=device, 
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        original_score_head = model.score.weight[0].float().detach().numpy()

        os.makedirs(f"{root}/{case_name}", exist_ok=True)
        torch.save(model.score.state_dict(), f"{root}/{case_name}/orignal-head.pth")
        print("save original head model:", f"{root}/{case_name}/orignal-head.pth")

        correlation_coefficients = [0 for i in range(2*k)]

        if args.full_composed:
            component_dir = f"{root}/{case_name}-{args.decompose_method}-component-full-composed"
        else:
            component_dir = f"{root}/{case_name}-{args.decompose_method}-component"

        
        os.makedirs(component_dir, exist_ok=True)
        print("save to:", component_dir)

        for i in range(k):
            component = components_vector[i][None,...]

            model.score.weight = torch.nn.Parameter(torch.Tensor(component).contiguous())
            if not args.debug:
                torch.save(model.score.state_dict(), f"{component_dir}/{case_name}-{args.decompose_method}-component{i}.pth")
                print("save model:", f"{component_dir}/{case_name}-{args.decompose_method}-component{i}.pth")

            model.score.weight = torch.nn.Parameter(torch.Tensor(-component).contiguous()) 
            if not args.debug:
                torch.save(model.score.state_dict(), f"{component_dir}/{case_name}-{args.decompose_method}-component{i+k}.pth")
                print("save model:", f"{component_dir}/{case_name}-{args.decompose_method}-component{i+k}.pth")

        



