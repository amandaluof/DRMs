base_model="Ray2333/Gemma-2B-rewardmodel-baseline" 
wandb_name="Gemma-2B-rewardmodel-baseline-emb-mixture2_and_safe_pku"
log_dir='reward_models_sft'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 accelerate launch --main_process_port 9996 cal_emb.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs 1 \
    --max_length 4096 \
    --gradient_accumulation_steps 64 \
    --learning_rate 5e-6 \
    --data_path 'weqweasdas/preference_dataset_mixture2_and_safe_pku' \
    --cls_embs_path /srv/local/ry21/own_data/${wandb_name} \
    

