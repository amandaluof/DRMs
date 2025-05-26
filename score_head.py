import os
import re
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


def extract_component_number(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'component(\d+)\.pth$', filename)
    if match:
        return int(match.group(1))
    else:
        return float('inf')

class MutipleHead(nn.Module):
    def __init__(self, hidden_size, output_size, score_head_weight, dtype, device, normalize_statistics_file=None, normalize=False, model_type=None, seed=None):
        super().__init__()
        self.device = device
        self.score_head = nn.Linear(hidden_size, output_size, dtype=dtype, bias=False) # model.config.hidden_size

        if score_head_weight is not None:
            # original implementation
            self.load_ckpts(score_head_weight, num_head=output_size)
        else:
            if seed is not None:
                print("seed for initialization:", seed)
                torch.manual_seed(seed)

            if model_type == "Random_v2":
                self.score_head = nn.Linear(hidden_size, output_size, dtype=dtype, bias=False)
            elif model_type == "Random_kaiming_uniform_L2norm":
                self.score_head.weight.data = (self.score_head.weight.float() / self.score_head.weight.float().norm(dim=-1, keepdim=True)).to(torch.bfloat16)
            elif model_type == "Random_gaussian_L2norm":
                self.score_head = nn.Linear(hidden_size, output_size, dtype=torch.bfloat16, bias=False)
                self.score_head.weight.data = torch.randn_like(self.score_head.weight, dtype=torch.float32)
                self.score_head.weight.data /= self.score_head.weight.data.norm(dim=-1, keepdim=True)
                self.score_head.weight.data = self.score_head.weight.data.to(torch.bfloat16)

        print("score_head.weight.shape:", self.score_head.weight.shape)
        print("weight head norm ==1: ", (self.score_head.weight.norm(dim=-1)==1).all())


        self.normalize = normalize
        if normalize:
            self.set_normalize_param(normalize_statistics_file)

    def load_ckpts(self, score_head_weight, num_head):        
        print('loading single reward model head')
        print("before load:", self.score_head.weight.sum())
        # score_head_weight给的是文件夹

        if os.path.exists(score_head_weight) and not os.path.isfile(score_head_weight):
            weight_dir = score_head_weight
            pth_files = glob.glob(os.path.join(weight_dir, "*.pth"))
            if len(pth_files) > 1 and pth_files[0] != "rng_state.pth":
                pth_files = sorted(pth_files, key=extract_component_number)
                num_pth = len(pth_files)
                pos_pth_files = pth_files[:int(num_pth/2)][:int(num_head/2)]
                neg_pth_files = pth_files[int(num_pth/2):][:int(num_head/2)]
                pth_files = pos_pth_files + neg_pth_files
#                print(len(pth_files))
                print("loaded the above weights")
                print("len(pth_files):", len(pth_files))

                weight_parts = []
                for pth_file in pth_files:
                    state = torch.load(pth_file, map_location="cpu")
                    # 假设每个 .pth 文件存储的是单个 tensor 或者存储在 state dict 中的 key "weight"
                    if isinstance(state, dict):
                        if "weight" in state:
                            weight = state["weight"]
                        else:
                            # 如果字典中没有明确的 "weight"，则假设字典本身就是权重矩阵
                            weight = state
                    else:
                        weight = state

                    weight_parts.append(weight)
                
                combined_weight = torch.cat(weight_parts, dim=0)
                
                # 打印调试信息
                print(f"加载了 {len(pth_files)} 个 .pth 文件，拼接后的权重 shape: {combined_weight.shape}")

                state['weight'] = combined_weight
                # print(combined_weight[0].sum())
                info = self.score_head.load_state_dict(state)
            else:
                safetensor_files = glob.glob(os.path.join(weight_dir, "*.safetensors"))
                if len(safetensor_files) >= 1:
                    head_tensors = {}
                    
                    path_list = [os.path.join(weight_dir, file) for file in safetensor_files]

                    for path in path_list:
                        with safe_open(path, framework="pt", device=self.device) as f:
                            for k in f.keys():
                                if k.startswith('score.'):
                                    head_tensors[k.replace("score.", "")] = f.get_tensor(k)

                    info = self.score_head.load_state_dict(head_tensors)
                    print("model.score_head")
                    print(info)
        
        print("model.score_head")
        print(info)
        print("after load:", self.score_head.weight.sum())

    def set_normalize_param(self, normalize_statistics_file):
        with open(normalize_statistics_file, "rb") as f:
            statistics = pickle.load(f)
        self.scores_mean = torch.Tensor(statistics["mean"][None,...]).to(self.device)
        self.scores_std = torch.Tensor(statistics["std"][None,...]).to(self.device)
        print("scores_mean:", self.scores_mean)
        print("scores_std:", self.scores_std)
        
    def __call__(self, embs):
        '''
        embs: (N, 2, 2048)
        '''
        num_sample = embs.shape[0]
        embs_chosen = embs[:, 0, :]
        embs_rejected = embs[:, 1, :]
        embs = torch.concat([embs_chosen, embs_rejected], 0).to(device=self.device, dtype=self.score_head.weight.dtype)

        scores = self.score_head(embs) # [2N, 1]
        rewards_chosen, rewards_rejected = scores[:num_sample],  scores[num_sample:]

        return rewards_chosen, rewards_rejected

