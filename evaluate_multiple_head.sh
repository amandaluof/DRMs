# step1. extract embedding of test set 
# step2. calculate scores with multiple score head 
# step3. TTA

root_path="/srv/local/ry21/own_code/reward_train_huan"  
emb_path="/srv/local/ry21/own_data/test_emb/"

basemodel_name="Ray2333/Gemma-2B-rewardmodel-baseline"
model_arch="Gemma-2B-rewardmodel-baseline"
case_name="Gemma-2B-rewardmodel-baseline-emb-mixture2_and_safe_pku"


# step1, extract embedding of test set 
export PYTHONPATH=$PYTHONPATH:"./reward-bench"
batch_size=1
if [[ "$case_name" == *"Llama"* ]]; then
    chat_template="tulu"
else
    chat_template="gemma"
fi

# rewardBench
python cal_emb_rewardBench.py \
    --model=$basemodel_name \
    --chat_template=$chat_template \
    --batch_size=${batch_size} \
    --not_quantized \
    --do_not_save \
    --emb_path $emb_path 

# rpr_multi_class
batch_size=1
rpr_max_len=3000
python3 cal_emb_custom.py \
    --per_device_eval_batch_size 1 \
    --max_length ${rpr_max_len} \
    --base_model $basemodel_name \
    --log_dir "rpr_multi_class_test/${case_name}" \
    --task "rpr_multi_class_test" \
    --cls_embs_path /srv/local/ry21/own_data/test_emb/rpr_multi_class_test_${case_name}

cd ./reward-bench

# step2, get performance of DRM heads
model_type="DRM"
num_head=100
score_head_weight="/srv/local/ry21/own_code/reward_models_sft/Gemma-2B-rewardmodel-baseline-emb-mixture2_and_safe_pku-PCA-component"

python3 ../evaluate_multiple_head.py \
            --model=$basemodel_name \
            --chat_template=$chat_template \
            --not_quantized \
            --emb_path $emb_path"/rewardBench_${model_arch}.pkl" \
            --task "rewardBench" \
            --score_head_type $model_type \
            --num_head $num_head \
            --score_head_weight $score_head_weight 


python3 ../evaluate_multiple_head.py \
        --model=$basemodel_name \
        --chat_template=$chat_template \
        --not_quantized \
        --res_csv "../data_sample_${model_arch}_rpr_multi_class_test.csv" \
        --task "rpr_multi_class_test" \
        --score_head_type $model_type \
        --num_head 100 \
        --score_head_weight=$score_head_weight 

# # rewardbench
cd ..

model_type="DRM"
num_eval_head=100
python3 inf_TTA_v2.py --task rewardBench --result_path "/srv/local/ry21/own_code/DRMs/reward-bench/results/eval-set-scores/Gemma-2B-rewardmodel-baselineDRMhead_100" --score_head_type $model_type --num_eval_head $num_eval_head --category_based --wo_original_head 

python3 inf_TTA_v2.py --task rpr_multi_class_test --result_path "/srv/local/ry21/own_code/reward_train_huan/rpr_multi_class_alignment/Gemma-2B-rewardmodel-baseline/DRM/rpr_multi_class_head_scores_num_head_100.json" --score_head_type $model_type --num_eval_head $num_eval_head --wo_original_head 

