import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def get_metric(args):
    
    def compute_metrics(eval_preds):
        metric = dict()

        # 예측값과 레이블 추출
        logits = eval_preds.predictions  # shape: [B]
        labels = eval_preds.label_ids    # shape: [B]

        labels = (labels > 0.5).astype(int)

        if args.is_submission or args.split_valid_by_paragraph:             #save logits for submission
            if args.local_rank == 0:
                np.save(f'./ckpt/{args.save_dir}/{args.save_name}.npy', logits)
                np.save(f'./ckpt/{args.save_dir}/{args.save_name}_label.npy', labels)
            metric['accuracy'] = 1.0
        else:
            # accuracy 계산 # 1 if >0.5 else 0
            preds = np.argmax(logits, axis=1) if logits.ndim > 1 else (logits > 0.).astype(int)
            metric['accuracy'] = accuracy_score(labels, preds)  

            # f1 score 계산
            metric['f1_score'] = f1_score(labels, preds, average='macro')

            # precision 계산
            metric['precision'] = precision_score(labels, preds, average='macro')

            # recall 계산
            metric['recall'] = recall_score(labels, preds, average='macro')

            # roc_auc score 계산
            metric['roc_auc'] = roc_auc_score(labels, logits)
        
        return metric

    return compute_metrics

