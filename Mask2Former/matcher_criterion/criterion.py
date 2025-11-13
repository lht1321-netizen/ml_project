#이제 마지막으로 mathcer에서 고른 query의 loss을 계산

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .matcher import hungarian_matcher
"""
생성자 파라미터 설명
oversample_ratio: uncertain 좌표를 고를 때 후보가 되는 좌표의 개수를 결정하는 비율  
important_sample_ratio: uncertain한 좌표를 몇개를 할 지 결정하는 비율
class_cost_weight: class cost의 weight
mask_cost_weight: mask cost의 weight
dice_cost_weigth: dice cost의 weight
eos_weight: class가 아닌 쿼리 즉 배경을 가리키는 쿼리의 가중치
K: 전체 좌표 중에서 dice와 mask loss를 연산할 때 사용하는 좌표의 수 
"""
class criterion(nn.Module):
    def __init__(self, oversample_ratio: float ,important_sample_ratio: float,
                class_cost_weight: float=2.0, mask_cost_weight: float=5.0, dice_cost_weight: float=5.0, eos_weigt: float=0.1,
                K: int=112*112 ):
        super().__init__()
        
        self.important_sample_ratio = important_sample_ratio
        self.oversample_ratio = oversample_ratio

        self.w_m = mask_cost_weight
        self.w_c = class_cost_weight
        self.w_d = dice_cost_weight
        
        self.matcher = hungarian_matcher(class_cost_weight=self.w_c, mask_cost_weight=self.w_m, dice_cost_weight=self.w_d)

        self.w_e = eos_weigt

        background_ratio = torch.ones(2)
        background_ratio[-1] = self.w_e
        self.background_ratio = background_ratio

        self.K = K

        self.mask_loss_jit = torch.jit.script(mask_sigmoid_ce_loss)
        self.dice_loss_jit = torch.jit.script(dice_loss)
    
    #중요도 샘플링을 하기 위한 점수를 계산하는 함수
    #(N, K)
    def calc_uncertainty_score(self, sampled_predicted_mask):
        #애매한 픽셀(확률 0.5에 가까운)을 높은 점수를 주기 위해 -절댓값
        score = -(torch.abs(sampled_predicted_mask))
        return score
    
    #class loss 연산 함수
    def calc_class_loss(self, matched: list[tuple], predicted_class: Tensor):
        B, Q = predicted_class.size(0), predicted_class.size(1)
        idx = self.get_query_idx1(matched, Q)
        target_class = torch.ones(B*Q, dtype=torch.int64, device=predicted_class.device)
        target_class[idx] = 0

        class_loss = F.cross_entropy(predicted_class.reshape(B*Q, 2), target_class, self.background_ratio)
        output = {'class_loss': class_loss}
        return output
    
    def get_query_idx1(self, matched: list[tuple], Q:int):
        idx = torch.cat([i*Q + q for i, (q, _) in enumerate(matched)])
        return idx
    
    def get_query_idx2(self, matched: list[tuple]):
        batch_idx = torch.cat([torch.full_like(q, i) for i, (q, _) in enumerate(matched)])
        query_idx = torch.cat([q for (q, _) in matched])
        return batch_idx, query_idx 

    def calc_mask_dice_loss(self, matched: list[tuple], predicted_mask:Tensor, label: list[dict]):
        q_idx = self.get_query_idx2(matched)
        #(B, Q, H/4, W/4) -> (T, 1, H/4, W/4) -> T: 확인할 총 마스크 수
        predicted_mask = predicted_mask[q_idx].unsqueeze(1)
        
        t_idx = self.get_target_idx(matched)
        #(T, H, W)- -> (T, 1, H, W)
        target_mask = torch.cat([t['mask'] for t in label], dim=0)
        #순서대로 바꾸기
        target_mask = torch.index_select(target_mask, dim=0, index=t_idx).unsqueeze(1)

        with torch.no_grad():
            coords = self.get_coords(predicted_mask).unsqueeze(1)
            target_point = F.grid_sample(
                target_mask,
                coords,
                align_corners=False
            ).squeeze(1).squeeze(1)
        
        predict_point = F.grid_sample(
            predicted_mask,
            coords,
            align_corners=False
        ).squeeze(1).squeeze(1)
        loss = {
                'mask_loss': self.mask_loss_jit(predict_point, target_point),
                'dice_loss': self.dice_loss_jit(predict_point, target_point)
            }

        return loss

    def get_target_idx(self, matched: list[tuple]):
        target = [t for (_, t) in matched]
        size = [t.numel() for t in target]

        idx = torch.cat(target)
        offset = torch.tensor([0] + list(torch.cumsum(torch.tensor(size[:-1]), 0)))
        offset = torch.repeat_interleave(offset, torch.tensor(size))

        return idx + offset

    def get_coords(self, predicted_mask):
        assert self.oversample_ratio >= 1, "oversample_ratio는 1보다 커야 합니다."
        assert 0 <= self.important_sample_ratio <= 1, "important_sample_ratio는 0과 1 사이이어야 합니다."
        
        n_sample = int(self.oversample_ratio * self.K)
        n_important = int(self.important_sample_ratio * self.K)
        n_random = self.K - n_important

        T = predicted_mask.shape[0]
        
        #(T, n_sample, 2)의 램덤 좌표
        #전체 예측에서 중요도를 계산해 뽑는 것이 아니라 램덤으로 뽑은 샘플에서 중요도 계산해서 뽑기  
        sample_coord = torch.rand(T, n_sample, 2, device=predicted_mask.device)*2 - 1
        
        #(T, n_sample, 2) -> (T, 1, n_sample, 2): grid_sample 형식에 맞추기 위해
        sample_coord_grid = sample_coord.unsqueeze(1)
        #(T, 1, 1, n_sample): 샘플로 뽑은 램던 좌표에서 값을 가지고 옴
        sample_point = F.grid_sample(
            predicted_mask,
            sample_coord_grid,
            align_corners=False
        )
        
        #(T, n_sample)
        sample_point = sample_point.squeeze(1).squeeze(1)
        #중요도 계산: (T, n_sample)
        score = self.calc_uncertainty_score(sample_point)
        #가장 중요도 높은 인덱스 n_important 가지고 옴(각 T별로) ex [1,2,0], [2,0,1], ...
        
        idx = torch.topk(score, k=n_important, dim=1)[1]
        # [0, n_sample, 2*n_sample, ...]
        offset = torch.arange(T, dtype=torch.long, device=predicted_mask.device)*n_sample
        #(T, n_sample) + (T, 1) -> (T, n_sample) + (T, n_sample)
        #offset 더해서 flatten 했을 때의 index 얻기
        idx += offset.unsqueeze(-1)

        important_coord = sample_coord.view(-1, 2)[idx.view(-1), :].view(T, n_important, 2)

        if n_random > 0:
            random_coord = torch.rand(T, n_random, 2, device=predicted_mask.device)*2 -1
            final_coord = torch.cat([important_coord, random_coord], dim=1)
        else:
            final_coord = important_coord
        
        return final_coord

    def forward(self, trans_dec_outputs: dict, targets: list[dict] ):

        auxiliary_out = trans_dec_outputs['auxiliary_out']

        #matcher: [(쿼리 인덱스 탠서, 타겟 인덱스 탠서), ...] -> batch 개수만큼 투플 있는 리스트
        matched = self.matcher(trans_dec_outputs, targets, K=self.K)

        final_loss = {}
        final_loss.update(self.calc_class_loss(matched, trans_dec_outputs['predicted_class']))
        final_loss.update(self.calc_mask_dice_loss(matched, trans_dec_outputs['predicted_mask'], targets))
        
        for i, aux in enumerate(auxiliary_out):
            matched = self.matcher(aux, targets, K=self.K)
            temp = self.calc_class_loss(matched, aux['predicted_class'])
            temp.update(self.calc_mask_dice_loss(matched, aux['predicted_mask'], targets))
            temp = {k + f"_{i}": v for k, v in temp.items()}
            final_loss.update(temp)
        
        return final_loss
        
#mask의 loss를 계산하기 위한 함수(1개의 배치당)
#(N, K), (N,K)
def mask_sigmoid_ce_loss(mask_prob: Tensor, mask_target: Tensor):
    assert mask_prob.size(-1) == mask_target.size(-1), "예측과 타겟의 사이즈가 안 맞습니다."
    
    loss = F.binary_cross_entropy_with_logits(mask_prob, mask_target, reduction='none')

    return loss.mean(-1).sum() / mask_prob.size(0)

#dice의 loss를 계산하는 함수(1개의 배치당)
#(N, K), (N,K)
def dice_loss(mask_prob: Tensor, mask_target: Tensor):
    mask_prob = mask_prob.sigmoid()

    numerator = 2 * (mask_prob * mask_target).sum(-1)
    denominator = mask_prob.sum(-1) + mask_target.sum(-1)

    loss = 1-(numerator+1) / (denominator+1)

    return loss.sum() / mask_prob.size(0)



