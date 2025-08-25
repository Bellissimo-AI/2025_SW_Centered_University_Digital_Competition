export CUDA_VISIBLE_DEVICES=0,1,2,3

# 1부터 10까지 순회
for FOLD in $(seq 0 9); do
  echo "===== Running fold ${FOLD} ====="
  python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port=4122 train.py \
    --is_kfold true \
    --fold_idx ${FOLD} \
    --model_name 'kykim/funnel-kor-base' \
    --save_dir "./baseline_paragraph_10fold_${FOLD}" \
    --per_device_eval_batch_size 1 \
    --data_dir './data_oof' \
    --max_length 512 \
    --logging_steps 20 \
    --ckpt_path "./ckpt/baseline_paragraph_10fold_${FOLD}/checkpoint-3234" \
    --save_name "valid_fold${FOLD}_step3234" \
    --split_valid_by_paragraph true
done
