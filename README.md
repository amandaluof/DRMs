# Prepare
- Download rewardBench 
  git clone https://github.com/allenai/reward-bench.git

- Download test set for [rpr_five_class](https://huggingface.co/datasets/amandaa/rpr_five_class/tree/main)

# Extract feature embedding
sh cal_reward_extract_emb.sh

# Get reward head
sh generate_head.sh

# Evaluate multiple head and TTA
sh evaluate multiple_head_2B.sh

