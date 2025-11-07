
import torch
from torch import nn, Tensor
from torch.nn import functional as F

# position embedding을 위해 사용
import math
#pip install fvcore 필요
import fvcore.nn.weight_init as weight_init

# 1단계 : masked attention + add & norm
class masked_attention(nn.Module):

    #c_dim: feature dimension = query dimension
    #n_head: 멀티해더 수
    #논문에서 dropout이 0일 때가 퍼포먼스가 가장 좋았다고 해서 default를 0으로 설정
    def __init__(self, C_dim, n_head, dropout=0.0):

        super(masked_attention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(C_dim, n_head, dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(C_dim)
        
        #parameter 초기화, xavier_uniform_ 사용
        self.reset_parameters()

    def reset_parameters(self):
        for par in self.parameters():
            if par.dim() > 1:
                nn.init.xavier_uniform_(par)

    # query와 key에 위치 임베딩 더하기 위해 구현
    def add_pos_embedding(self, x: Tensor, pos: Tensor) -> Tensor:
        return x if pos is None else x + pos

    # x_last: 그전 단계의 출력값
    # img_feat: 이미지 feature map(C_dim으로 차원변환 된)
    # mask: 마스크 텐서 / 키 마스크는 없는 것으로 설정 -> input으로 넣기 전에 전처리 따로 할 예정
    # query_pos: 쿼리 포지션 임베딩 (쿼리가 어떤 위치를 나타내는지 알려주는 임베딩)
    # pixel_pos: 픽셀 포지션 임베딩 (이미지 픽셀의 위치 정보를 담은 임베딩)
    def forward(self, x_last, img_feat: Tensor,
                mask: Tensor,
                query_pos: Tensor, pixel_pos: Tensor
                ):
        # masked attention
        mask_attened = self.multihead_attn(query = self.add_pos_embedding(x_last, query_pos),
                                        key = self.add_pos_embedding(img_feat, pixel_pos),
                                        value = img_feat,
                                        attn_mask = mask,
                                        key_padding_mask = None
                                        )[0]
        # add & norm
        # 논문에 나오는 add&norm 부분을 masked attention 모듈 안에 구현
        # add: x + f(x) / norm : 정규화(LayerNorm)
        x_next = x_last + self.dropout(mask_attened)
        x_next = self.norm(x_next)

        return x_next

#2단계 : self attention + add & norm
class self_attention(nn.Module):
    #c_dim: feature dimension = query dimension
    #n_head: 멀티해더 수
    #논문에서 dropout이 0일 때가 퍼포먼스가 가장 좋았다고 해서 default를 0으로 설정
    def __init__(self, C_dim, n_head, dropout=0.0):
        super(self_attention, self).__init__()
        self.self_attn = nn.MultiheadAttention(C_dim, n_head, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(C_dim)

        self.reset_parameters()

    #parameter 초기화, xavier_uniform_ 사용
    def reset_parameters(self):
        for par in self.parameters():
            if par.dim() > 1:
                nn.init.xavier_uniform_(par)
    
    # query와 key에 위치 임베딩 더하기 위해 구현
    def add_pos_embedding(self, x: Tensor, pos: Tensor) -> Tensor:
        return x if pos is None else x + pos
    
    # x_last: 그전 단계의 출력값
    # mask: 마스크 텐서 / 키 마스크는 없는 것으로 설정(input으로 넣기 전에 전처리 따로 할 예정)
    # query_pos: 쿼리 포지션 임베딩 (쿼리가 어떤 위치를 나타내는지 알려주는 임베딩)
    # self attention이기 때문에 key = value
    def forward(self, x_last, mask = None, query_pos =None):
        # self attention
        Query_and_Key = self.add_pos_embedding(x_last, query_pos)
        
        self_attened = self.self_attn(
            query = Query_and_Key,
            key = Query_and_Key,
            value = x_last,
            attn_mask = mask,
            key_padding_mask = None
            )[0] 
        
        # add & norm
        # 논문에 나오는 add&norm 부분을 self attention 모듈 안에 구현
        # add: x + f(x) / norm : 정규화(LayerNorm)
        x_next = x_last + self.dropout(self_attened)
        x_next = self.norm(x_next)

        return x_next

#3단계 : FFN + add & norm
class FFN(nn.Module):
    #C_dim: feature dimension
    #dim_feedforward: FFN 내부의 hidden layer 차원
    #논문에서 dropout이 0일 때가 퍼포먼스가 가장 좋았다고 해서 default를 0으로 설정
    def __init__(self, C_dim, dim_feedforward=2048, dropout=0.0):
        super(FFN, self).__init__()
        
        self.first_linear = nn.Linear(C_dim, dim_feedforward)
        self.second_linear = nn.Linear(dim_feedforward, C_dim)
        
        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu

        self.norm = nn.LayerNorm(C_dim)
        #parameter 초기화, xavier_uniform_ 사용
        self.reset_parameters()

    def reset_parameters(self):
        for par in self.parameters():
            if par.dim() > 1:
                nn.init.xavier_uniform_(par)

    def forward(self, x_last):
        # FFN
        # FFN = relu(w1x + b1)W2 + b2
        ffn = self.second_linear(self.dropout(self.activation(self.first_linear(x_last))))

        # add & norm
        # 논문에 나오는 add&norm 부분을 FFN 모듈 안에 구현
        # add: x + f(x) / norm : 정규화(LayerNorm)
        x_next = x_last + self.dropout(ffn)
        x_next = self.norm(x_next)

        return x_next

#마스킹 위해 Multi-Layer Perceptron(MLP) 구현
class mlp(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim):
        super(mlp, self).__init__()
        
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.layers = self.make_layers()
        self.last_layer = nn.Linear(hidden_dim, output_dim)

    def make_layers(self):
        mlp_layers = [nn.Linear(self.input_dim, self.hidden_dim)]

        for _ in range(self.n_layers-2):
            mlp_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        return nn.ModuleList(mlp_layers)
    
    def forward(self, input):
        for layer in self.layers:
            input = F.relu(layer(input))
        
        output = self.last_layer(input)
        return output

#pixel position embedding을 하기 위한 모델 구현
#sinusoidal position embedding 사용
class pixel_position_embedding(nn.Module):
    def __init__(self, d_model):
        super(pixel_position_embedding, self).__init__()
        self.d_model = d_model

    #feature_map : (B, C, H, W)
    def forward(self, feature_map: Tensor):
        #sinusoidal을 사용하고 [H | W] 같이 H와 W를 나누어서 입력해야 해서 d_model이 4의 배수여야 함
        assert self.d_model % 4 == 0, "차원이 4의 배수가 아닙니다."
        
        #좌표 격자 만들기
        B, H, W = feature_map.size(0), feature_map.size(2), feature_map.size(3)
        H_coord, W_coord = torch.meshgrid(torch.arange(1, H+1, dtype=torch.float32, device=feature_map.device), 
                                          torch.arange(1, W+1, dtype=torch.float32, device=feature_map.device),
                                          indexing='ij')
        
        #좌표 스케일링 0~2ㅠ
        H_coord = H_coord.unsqueeze(0).expand(B, -1, -1) / (H + 1e-6) * (2 * math.pi)
        W_coord = W_coord.unsqueeze(0).expand(B, -1, -1) / (W + 1e-6) * (2 * math.pi)

        #pos 나눌 분모 만들기
        half_d = self.d_model // 2
        denom = torch.arange(0, half_d, dtype=torch.float32, device=feature_map.device) // 2
        denom = 10000 ** (2*denom / half_d)

        #pos/10000^(2i/d_model)
        pos_H = H_coord.unsqueeze(-1) / denom
        pos_W = W_coord.unsqueeze(-1) / denom

        #sin, cos 적용(짝수는 sin, 홀수는 cos)
        pos_H[:, :, :, 0::2].sin_()
        pos_H[:, :, :, 1::2].cos_()
        pos_W[:, :, :, 0::2].sin_() 
        pos_W[:, :, :, 1::2].cos_()

        #H와 W 합치기 및 (B, d_model, H, W)로 차원 변경
        pos_embedding = torch.cat([pos_H, pos_W], dim=-1).permute(0, 3, 1, 2)

        # output shape : (B, d_model, H, W)
        return pos_embedding
        
class Transformer_Decoder(nn.Module):
    """
    파라미터 설명란
    C_dim: feature dimension = query dimension
    n_head: 멀티헤더 어텐션의 헤더 수
    dim_feedforward: FFN 내부의 hidden layer 차원
    channel_dim_list: 각 feature map위 채널 차원수 리스트 [1/32, 1/16, 1/8, 1/4]
    mask_dim : 1/4 해상도의 C 차원
    L: 디코더 반복 횟수 따라서 총 3L개의 레이어가 있다.(논문에 3을 사용하여 default를 3으로 설정)
    dropout: 드롭아웃 (논문에서 0일때가 퍼포먼스가 가장 좋았다고 해서 default를 0으로 설정)
    n_query: 쿼리의 수 (논문에서 100을 사용하여 default를 100으로 설정)
    n_class: 구분해야 하는 클래수 수(건물 하나의 클래스만 할 것이므로 default 1로 설정)
    """
    def __init__(self, 
                C_dim: int,
                n_head: int,
                dim_feedforward: int,
                channel_dim_list: list,
                mask_dim: int,
                L: int = 3,
                dropout: float = 0.0,
                n_query: int = 100,
                n_class: int = 1):
        super(Transformer_Decoder, self).__init__()
        
        #기본 파라미터 저장
        self.channel_dim_list = channel_dim_list
        self.L = L
        self.C_dim = C_dim
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.mask_dim = mask_dim
        self.n_class = n_class

        # 디코더 레이어 만들기, 3개 층을 L번 반복
        self.masked_attn_layers = nn.ModuleList()
        self.self_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.make_layers()

        #쿼리 임베딩 만들기(학습가능 임베딩으로 만들기)
        self.n_query = n_query
        self.query_feat_embed = nn.Embedding(n_query, C_dim)
        self.query_pos_embed = nn.Embedding(n_query, C_dim)

        #해상도 임베딩 만들기(3가지 해상도에 대해 학습가능 임베딩으로 만들기)
        self.resolution_embed = nn.Embedding(3, C_dim)
        
        #각 feature map의 채널과 C_dim이 다를 때 차원을 변환하는 모듈 만들기
        self.projection_modules = self.make_projection_modules()

        #픽셀 위치 임베딩 모듈 만들기
        self.pixel_pos_embedding = pixel_position_embedding(d_model=C_dim)


        self.Cdim_norm = nn.LayerNorm(C_dim)

        #class + 배경 백터
        self.class_predic = nn.Linear(C_dim, self.n_class+1)
        #mask embed: maske_dim(1/4 feature map의 C)로 바꿈
        self.mask_embed = mlp(n_layers=3, input_dim=self.C_dim,
                              hidden_dim=self.C_dim, output_dim=self.mask_dim)


    
    def make_layers(self):
        for _ in range(self.L * 3):
            self.masked_attn_layers.append(masked_attention(
                C_dim = self.C_dim,
                n_head = self.n_head,
                dropout = self.dropout
                ))
            self.self_attn_layers.append(self_attention(
                C_dim = self.C_dim,
                n_head = self.n_head,
                dropout = self.dropout
                ))
            self.ffn_layers.append(FFN(
                C_dim = self.C_dim,
                dim_feedforward = self.dim_feedforward,
                dropout = self.dropout
                ))
    
    def make_projection_modules(self):
        output = nn.ModuleList()
        for channel_dim in self.channel_dim_list:
            if channel_dim != self.C_dim:
                output.append(nn.Conv2d(channel_dim, self.C_dim, kernel_size=1))
                weight_init.c2_xavier_fill(output[-1])
            else:
                output.append(nn.Identity())
        return output
    
    def predic_class_mask_masking(self, x, feat_map4: Tensor, masking_size):
        x_normed = self.Cdim_norm(x)
        class_predic = self.class_predic(x_normed)
        mask_embed = self.mask_embed(x_normed)
        
        # 1/4 해상도 feature map과 mask embed 내적
        # (B, Q, C) @ (B, C, H/4, W/4) -> (B, Q, H/4, W/4)
        mask_predic = torch.einsum("bqc, bchw -> bqhw", mask_embed, feat_map4)

        #(B, Q, H_i, W_i) -> (B, Q, H_i*W_i)
        masking = F.interpolate(mask_predic, size=masking_size, mode='bilinear', align_corners= False).sigmoid().flatten(2)
        masking = masking.unsqueeze(0).repeat(self.n_head, 1, 1, 1).flatten(0,1)
        #0.5이상이면 F 미만이면 T
        #T이면 마스킹을 해라, F면 attention을 하라
        masking = (masking < 0.5).bool()
        #masking이 mask_predic으로 부터 파생되는 값이므로 detach 실행
        masking = masking.detach()

        return class_predic, mask_predic, masking

    #feature_pyramid: 픽셀 디코더에서 만든 [1/32, 1/16, 1/8, 1/4] 해상도의 feature map 리스트
    #각 feature map은 (B, C, H, W) 형태
    def forward(self, feature_pyramid: list[Tensor]):
        #feature map의 채널 차원 확인
        assert [feature_pyramid[i].size(1) for i in range(len(feature_pyramid))] == self.channel_dim_list, "feature map의 채널 차원이 맞지 않습니다."
        
        #쿼리에 맞게 차원 변환된 feature map + 해상도 embed
        proj_feat_list = []
        #pixel에 따른 embed 리스트(상수)
        pos_emb_list = []
        #해상도 크기 리스트 (H, W)
        HW_list = []
        #배치사이즈
        B = feature_pyramid[0].size(0)

        for i in range(3):
            #(B, C, H, W) -> (B, C_dim, HxW) + (1, C_dim, 1) -> (B, C_dim, HxW)
            proj_feat_list.append(
                self.projection_modules[i](feature_pyramid[i]).flatten(2)
                + self.resolution_embed.weight[i][None, :, None])
            #(B, C_dim, HxW)->(B, HxW, C_dim)
            proj_feat_list[-1] = proj_feat_list[-1].permute(0, 2, 1)
            
            #(B, C_dim, H, W) -> (B, C_dim, HxW)->(B, HxW, C_dim)
            pos_emb_list.append(self.pixel_pos_embedding(feature_pyramid[i]).flatten(2).permute(0, 2, 1))
            #HW의 int값을 가지는 리스트
            HW_list.append(feature_pyramid[i].shape[-2:])

        # (Q, C_dim)-> (B, Q, C_dim)
        # layer에 입력할 쿼리에 layer에서 더할 쿼리 위치 embed 설정
        x = self.query_feat_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        query_pos_embed = self.query_pos_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        class_predic_list = []
        mask_predic_list = []

        class_predic, mask_predic, masking = self.predic_class_mask_masking(x, feat_map4=feature_pyramid[-1], masking_size=HW_list[0])

        class_predic_list.append(class_predic)
        mask_predic_list.append(mask_predic)

        #총 3L 레이어 : (1/32->1/16->1/8)*3
        for i in range(self.L):
            for j in range(3):
                masking[torch.where(torch.all(masking, dim=-1))] = False
                
                index = 3*i + j
                #mask attention layer
                x = self.masked_attn_layers[index](
                    x_last=x,
                    img_feat=proj_feat_list[j],
                    mask = masking,
                    query_pos = query_pos_embed,
                    pixel_pos = pos_emb_list[j]
                    )
                #self attention layer
                x = self.self_attn_layers[index](
                    x_last=x,
                    query_pos=query_pos_embed,
                    mask = None
                )
                #FFN layer
                x = self.ffn_layers[index](
                    x_last=x
                )
                
                # 다음 단계 마스킹 예상 + 마스크, 클래스 예상
                class_predic, mask_predic, masking = self.predic_class_mask_masking(x, feat_map4=feature_pyramid[-1], masking_size=HW_list[(j+1)%3])
                
                #결과 리스트에 추가
                class_predic_list.append(class_predic)
                mask_predic_list.append(mask_predic)

        assert len(class_predic_list) == 3*self.L + 1, "예측 클래스 개수 안 맞음"
        assert len(mask_predic_list) == 3*self.L + 1, "예측 마스크 개수 안 맞음"

        final_out = {
                'predicted_class' : class_predic_list[-1],
                'predicted_mask' : mask_predic_list[-1],
                'auxiliary_out' : [
                                   {"predicted_class":c, "predicted_mask":m}
                                   for c, m in zip(class_predic_list[:-1], mask_predic_list[:-1])                   
                                  ]
                                  
                    }
        return final_out










        




