import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRloss(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) loss

    * logits  : 1-D tensor [batch] -– 각 샘플의 예측 score
    * labels  : 1-D tensor [batch] -– binary (1 = positive, 0 = negative)

    미니배치 안에 **최소 1개 이상의 positive / negative** 가 모두 있어야 합니다.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps          # log(0) 방지용 작은 값

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # ① positive / negative 분리
        pos_scores = logits[labels == 1]          # [P]
        neg_scores = logits[labels == 0]          # [N]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            # 배치 안에 한쪽이 없다면 BPR 을 계산할 수 없음
            return logits.new_tensor(0.0, requires_grad=True)

        # ② 모든 양-음 쌍의 차이 s_i − s_j  계산
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)   # [P, N]

        # ③ BPR:  -log σ(s_i − s_j)
        loss = -torch.log(torch.sigmoid(diff) + self.eps).mean()
        return loss