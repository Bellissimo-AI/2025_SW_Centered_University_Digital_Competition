# 2025 SW-Centered University Digital Competition: AI Sector

## Awards
- **3rd Place (Excellence Award)** *Team SKKU AI*  
- **President’s Award** — Institute for Information & Communications Technology Planning & Evaluation (IITP)  
- [Competition link](https://dacon.io/competitions/official/236473/overview/description)  

---

## Experimental environment
- **Hardware**: NVIDIA A100 80GB × 8 GPUs  
- **Storage**: At least 200GB of free disk space recommended  
- **Software**: Python 3.10  
- For additional dependencies, please refer to `requirements.txt`.  

---

## Overview
- **Phase 1** is trained using document-level labels.  
- The trained Phase 1 model is then used to generate paragraph-level pseudo-labels for each validation set.  
- **Phase 2** is trained on these paragraph-level pseudo-labels.  
- The trained Phase 2 model is again used to refine paragraph-level pseudo-labels.  
- **Phase 3** is trained on this refined set of paragraph-level pseudo-labels.  
- **Phase 4** extends Phase 3 by incorporating additional data augmented with a large language model (LLM).  

---

## Training
```bash
# Phase 1 training and evaluation
sh scripts_final/train_phase1_10fold_funnel_kor_base.sh
sh scripts_final/eval_validfold_phase1_funnel_kor_base.sh

# Merge validation results and generate Phase 2 data
# Run notebooks/merge_phase1_oof.ipynb (ensure the correct path)

# Phase 2 training and evaluation
sh scripts_final/train_phase2_10fold_funnel_kor_base.sh
sh scripts_final/eval_validfold_phase2_funnel_kor_base.sh

# Merge validation results and generate Phase 3 data
# Run notebooks/merge_phase2_oof.ipynb (ensure the correct path)

# Phase 3 training
sh scripts_final/train_phase3_10fold_bert_kor_base.sh

# LLM-based data augmentation
# Run notebooks/LLM_generation.ipynb
# Run notebooks/merge_phase3_with_LLM.ipynb (ensure the correct path)

# Phase 4 training
sh scripts_final/train_phase4_10fold_bert_kor_base.sh
```

---
## Inference
```bash
eval.sh

#fold별 추론 결과 병합하기
#notebooks/inference_merge_kfold.ipynb 실행 (경로를 맞춰주세요)
```
