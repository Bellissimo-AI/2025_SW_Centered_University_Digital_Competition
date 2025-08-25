#bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#Leaderbord 성능은 대회 종료시까지 10개 폴드가 다 돌아가지 않아 0~7번 폴드까지만 사용하였습니다.

#-------------------------------------- phase 4 -----------------------------------------------------#
for FOLD in $(seq 0 7); do
  echo "===== Running fold ${FOLD} ====="
  python -m torch.distributed.launch --nproc_per_node=8 --use_env --master_port=12412 train.py \
    --is_kfold true \
    --fold_idx ${FOLD} \
    --model_name 'kykim/bert-kor-base' \
    --save_dir "./kykim_phase4_${FOLD}" \
    --per_device_eval_batch_size 1 \
    --data_dir './data' \
    --max_length 512 \
    --ckpt_path "./ckpt/kykim_phase4_${FOLD}/checkpoint-9904" \
    --save_name "test_fold${FOLD}_step9904" \
    --is_submission true
done
#--------------------------------------------------------------------------------------------------#


#-------------------------------------- phase 3 -----------------------------------------------------#
#마찬가지로 Leaderbord 성능은 대회 종료시까지 10개 폴드가 다 돌아가지 않아 0~3번 폴드까지만 사용하였습니다.
for FOLD in $(seq 0 3); do
  echo "===== Running fold ${FOLD} ====="
  python -m torch.distributed.launch --nproc_per_node=8 --use_env --master_port=12412 train.py \
    --is_kfold true \
    --fold_idx ${FOLD} \
    --model_name 'kykim/bert-kor-base' \
    --save_dir "./kykim_phase3_${FOLD}" \
    --per_device_eval_batch_size 1 \
    --data_dir './data' \
    --max_length 512 \
    --ckpt_path "./ckpt/kykim_phase3_${FOLD}/checkpoint-6468" \
    --save_name "test_fold${FOLD}_step6468" \
    --is_submission true
done
#--------------------------------------------------------------------------------------------------#