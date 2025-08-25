#!/usr/bin/env bash

# 사용할 GPU 목록
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 1부터 10까지 순회
for FOLD in $(seq 0 9); do
  echo "===== Running fold ${FOLD} ====="
  python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py \
    --is_kfold true \
    --k_fold 10 \
    --fold_idx ${FOLD} \
    --model_name 'kykim/funnel-kor-base' \
    --save_dir "./baseline_paragraph_phase3_10fold_${FOLD}" \
    --data_dir './data_oof_phase3' \
    --max_length 256 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 10 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.1 \
    --weight_decay 1e-3 \
    --logging_steps 20
done