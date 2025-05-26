base_model="Ray2333/Gemma-2B-rewardmodel-baseline" 
wandb_name=("Gemma-2B-rewardmodel-baseline-emb-mixture2_and_safe_pku")

python3 generate_head.py --basemodel_names $base_model --case_names $wandb_name --decompose_method "PCA"  --num_component 50