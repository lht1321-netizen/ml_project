import torch
from torch import nn, Tensor
import torch.nn.functional as F
#spicy 패키지 다운 필요
from scipy.optimize import linear_sum_assignment


class hungarian_matcher(nn.Module):
    #각각의 loss의 가중치들을 파라미터로 가짐
    def __init__(self, class_cost_weight:float = 1, mask_cost_weight:float = 1, dice_cost_weight:float = 1):
        super().__init__()
    
        self.weight_c = class_cost_weight
        self.weight_m = mask_cost_weight
        self.weight_d = dice_cost_weight
        assert not (self.weight_c == 0 and self.weight_d == 0 and self.weight_m == 0), "가중치가 모두 0입니다."
        
        # just in time compiler 이용 계산 최적화
        self.mask_sigmoid_ce_loss_jit = torch.jit.script(query_mask_sigmoid_ce_loss)
        self.dice_loss_jit = torch.jit.script(query_dice_loss)
    
    """
    Trans_dec_oupts: Transform Decoder에서 만든 최종 output
    {
    'predicted_class' : (B, Q, 2)
    'predicted_mask' : (B, Q, H/4, W/4),
    'auxiliary_out' : [{"predicted_class":c, "predicted_mask":m}
                            for c, m in zip(class_predic_list[:-1], mask_predic_list[:-1])                   
                      ]
    }

    targets: 이미지의 라벨 [{instance_class: tensor, mask: tensor[N,H,W]}, {...} , {...}, ...]
    K: 램덤으로 확인할 좌표(픽셀) 수 -> 논문에서 112*112를 사용해서 default 값으로 사용
    """
    def forward(self, trans_dec_outputs: dict, targets: list[dict] , K: int = 112*112):
        
        B, Q = trans_dec_outputs['predicted_class'].shape[0:2]
        assert B == trans_dec_outputs['predicted_mask'].shape[0] and Q == trans_dec_outputs['predicted_mask'].shape[1], "B랑 Q가 이상합니다."

        matched_idx = []
        for i_b in range(B):
            #(Q, 2)
            class_prob = trans_dec_outputs['predicted_class'][i_b].softmax(-1)
            #(N)
            target_class = targets[i_b]['instance_class'].to(class_prob.device)
            #(Q, N=1 or 0) 존재하는 클라스가 1개이므로 하나의 사진에 있을 수 있는 클라스는 0 or 1 
            class_cost = -class_prob[:, target_class]

            #(Q, H/4, H/4) -> (Q, 1, H/4, W/4)
            mask_prob = trans_dec_outputs['predicted_mask'][i_b].unsqueeze(1)
            #(N, H, W) -> (N, 1, H, W)
            target_mask = targets[i_b]['mask'].to(mask_prob).unsqueeze(1)
            
            # [-1, 1) 사이 램덤 좌표 K개 생성 (1, 1, K, 2)
            rand_coords = torch.rand(1, 1, K, 2, device=class_prob.device)*2 -1

            #random 하게 생성된 좌표에 해당하는 mask_prob 값 가져오기        
            mask_prob = F.grid_sample(
                mask_prob,
                #(Q, 1, K, 2)
                rand_coords.repeat(mask_prob.shape[0], 1, 1, 1),
                align_corners = False
            )
            #(Q, 1, 1, K) -> (Q,K)
            mask_prob = mask_prob.squeeze(1).squeeze(1)

            #random 하게 생성된 좌표에 해당하는 target_mask 값 가져오기
            target_mask = F.grid_sample(
                target_mask,
                #(Q, 1, K ,2)
                rand_coords.repeat(target_mask.shape[0], 1, 1, 1),
                align_corners = False
            )
            #(N, 1, 1, K) -> (N, K)
            target_mask = target_mask.squeeze(1).squeeze(1)

            #float 32로 연산
            with torch.cuda.amp.autocast(enabled=False):
                mask_prob = mask_prob.float()
                target_mask = target_mask.float()
                # mask match 비용 계산
                mask_cost = self.mask_sigmoid_ce_loss_jit(mask_prob, target_mask, K)
                # dice match 비용 계산
                dice_cost = self.dice_loss_jit(mask_prob, target_mask, K)

            weighted_cost = self.weight_m*mask_cost + self.weight_d*dice_cost + self.weight_c*class_cost
            weighted_cost = weighted_cost.cpu()
            matched_idx.append(linear_sum_assignment(weighted_cost.detach().numpy()))

        #[(쿼리 인덱스 탠서, 타겟 인덱스 탠서), ...] -> batch 개수만큼 투플 있는 리스트
        return [(torch.as_tensor(p, dtype=torch.int64), torch.as_tensor(t, dtype=torch.int64))
                    for p, t in matched_idx]

def query_mask_sigmoid_ce_loss(mask_prob: Tensor, target_mask: Tensor, K: int):
            assert mask_prob.shape[-1]==K and target_mask.shape[-1]==K, "K가 안 맞습니다."

            #정답이 1일 때의 loss 계산 -> (Q, K)
            calc_one_loss = F.binary_cross_entropy_with_logits(
                mask_prob, torch.ones_like(mask_prob), reduction='none'
            )
            #정답이 0일 때의 loss 계산 -> (Q, K)
            calc_zero_loss = F.binary_cross_entropy_with_logits(
                mask_prob, torch.zeros_like(mask_prob), reduction='none'
            )

            loss_sum = (
                        #모두 1일때의 loss 값과 실제 마스크를 곱해 실제 1일 때의 loss 구함
                        #(Q, K) @ (K, N) -> (Q,  N)
                        torch.einsum('qk,nk->qn', calc_one_loss, target_mask)
                        #모두 0일때의 loss 값과 실제 마스크를 곱해(0과 1을 바꾼) 실제 0일때의 loss 구함
                        #(Q, K) @ (K, N) -> (Q,  N)
                        + torch.einsum('qk,nk->qn', calc_zero_loss, (1 - target_mask))
                        )
            return loss_sum / K

def query_dice_loss(mask_prob: Tensor, target_mask: Tensor, K: int):
            assert mask_prob.shape[-1]==K and target_mask.shape[-1]==K, "K가 안 맞습니다."
            assert mask_prob.ndim == target_mask.ndim == 2 , "차원이 안 맞습니다."
            
            mask_prob = mask_prob.sigmoid()
            #2(A^B)
            numerator = 2*(torch.einsum('qk,nk->qn', mask_prob, target_mask))
            #|A|+|B|
            denominator = mask_prob.sum(-1).unsqueeze(-1) + target_mask.sum(-1).unsqueeze(0)
            # 1- 2(A^B)/(|A|+|B|) + zero division error 방지
            loss = 1 -(numerator+1)/(denominator+1)
            return loss

        







            