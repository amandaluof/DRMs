# Rethinking Diverse Human Preference Learning through Principal Component Analysis

This script provides the pipeline to extract embeddings, generate reward heads, and evaluate performance on [RewardBench](https://github.com/allenai/reward-bench) and the `rpr_five_class` dataset.


# Prepare
- Download rewardBench 
  ```bash
  git clone https://github.com/allenai/reward-bench.git
  ```
- Download test set for [rpr_five_class](https://huggingface.co/datasets/amandaa/rpr_five_class/tree/main)
  
# Extract feature embedding
```bash
sh cal_reward_extract_emb.sh
```
# Get reward head
```bash
sh generate_head.sh
```
# Evaluate multiple head and TTA
```bash
sh evaluate multiple_head.sh
```
# Citations
If you use this pipeline or `rpr_five_class` dataset in your research, please cite:

```bibtex
@article{luo2025rethinking,
  title={Rethinking diverse human preference learning through principal component analysis},
  author={Luo, Feng and Yang, Rui and Sun, Hao and Deng, Chunyuan and Yao, Jiarui and Shen, Jingyan and Zhang, Huan and Chen, Hanjie},
  journal={arXiv preprint arXiv:2502.13131},
  year={2025}
}
```