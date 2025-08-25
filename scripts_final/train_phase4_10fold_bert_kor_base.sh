#!/usr/bin/env bash

# 사용할 GPU 목록
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# 1부터 10까지 순회
for FOLD in $(seq 0 9); do
  echo "===== Running fold ${FOLD} ====="
  python -m torch.distributed.launch --nproc_per_node=6 --use_env --master_port=12456 train.py \
    --is_kfold true \
    --k_fold 10 \
    --fold_idx ${FOLD} \
    --model_name 'kykim/bert-kor-base' \
    --save_dir "./kykim_phase4_${FOLD}" \
    --data_dir './data_oof_phase4' \
    --max_length 256 \
    --per_device_train_batch_size 42 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.1 \
    --weight_decay 1e-3 \
    --logging_steps 20
done