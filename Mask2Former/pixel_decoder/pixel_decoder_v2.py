import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_
import fvcore.nn.weight_init as weight_init

import math
import copy
from ..transformer_decoder.transformer_decoder import pixel_position_embedding
from .ops.modules.ms_deform_attn import MSDeformAttn

#positional embedding 더하는 함수
def add_pos_embedding(x: Tensor, pos: Tensor) -> Tensor:
        return x if pos is None else x + pos

class MSDeform_attn_transformer_encoder_layer(nn.Module):
    def __init__(self, d_model: int=256, d_ffn:int=1024,
                 dropout: float=0.1, n_heads: int=8, n_points: int=4):
        super().__init__()
        #MSDeformAttn는 c와 c++로 이루어진 clas(파이토치로 구현 불가)
        self.msd_self_attn = MSDeformAttn(d_model, n_levels=3, n_heads=n_heads, n_points=n_points)
        self.dropout_1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        #ffn layer
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu
        self.dropout_2 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout_3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, input, pos_embed, ref_points, level_shapes, level_start_idx):
        #msd self attention
        fx_self = self.msd_self_attn(query=add_pos_embedding(input, pos_embed),
                                       reference_points=ref_points,
                                       input_flatten=input,
                                       input_spatial_shapes=level_shapes,
                                       input_level_start_index=level_start_idx,
                                       input_padding_mask=None
                                    )
        fx_plus_x_self = input + self.dropout_1(fx_self)
        fx_plus_x_self = self.norm1(fx_plus_x_self)
        
        #ffn layer
        fx_ffn = self.linear2(self.dropout_2(self.activation(self.linear1(fx_plus_x_self))))
        fx_plus_x_ffn = fx_plus_x_self + self.dropout_3(fx_ffn)
        fx_plus_x_ffn = self.norm2(fx_plus_x_ffn)

        return fx_plus_x_ffn

class MSDeform_attn_transformer_encoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, n_encoder_layers: int):
        super().__init__()
        self.n_encoder_layers = n_encoder_layers
        #n_encoder_layers 수 만큼의 MSDeform_attn_transformer_encoder_layer 모듈 리스트 만들기
        self.encoder = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(self.n_encoder_layers)])

    def forward(self, input, level_shapes, level_start_idx, pos_embed, B):
        output = input
        ref_points = self.make_ref_points(level_shapes=level_shapes, input_device=input.device, B=B)
        for L in self.encoder:
            output = L(output, pos_embed, ref_points, level_shapes, level_start_idx)
        
        return output
    
    @staticmethod
    # reference 좌표 만들기
    # 사진의 모든 영역이 유효하다는 가정에서 만듬(패딩 없음)
    def make_ref_points(level_shapes, input_device, B):
        ref_points_list = []
        L = len(level_shapes)
        for level, (h, w) in enumerate(level_shapes):
            #각 픽셀의 중심을 가리키는 좌표 생성
            y_points, x_points = torch.meshgrid(torch.linspace(0.5, h-0.5, h, dtype=torch.float32, device=input_device),
                                                torch.linspace(0.5, w-0.5, w, dtype=torch.float32, device=input_device),
                                                indexing='ij')
            #flatten 후 정규화 [0,1]
            # (1, hxw)
            y_points = y_points.reshape(-1).unsqueeze(0) / h
            x_points = x_points.reshape(-1).unsqueeze(0) / w
            #(1, hxw, 2)
            level_ref_points = torch.stack((x_points, y_points), -1)
            #(B, hxw, 2)
            level_ref_points = level_ref_points.expand(B, -1, -1)
            #[(B, hxw, 2), ...]
            ref_points_list.append(level_ref_points)
        
        #(B, h1xw1, 2) + (B, h2xw2, 2) +.... -> (B, Total_pixel, 2)
        level_merge = torch.cat(ref_points_list, 1)
        #(B, T, 2) -> (B, T, L, 2) 
        level_merge = level_merge.unsqueeze(2).expand(-1, -1, L, -1)

        return level_merge
             
class MSDeform_attn_transformer_encoder_only(nn.Module):
    def __init__(self, d_model=256, n_head=8, n_encoder_layers=6, 
                 d_ffn=1024, dropout=0.1, n_points=4):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.n_encoder_layers = n_encoder_layers
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.n_points = n_points

        encoder_layer = MSDeform_attn_transformer_encoder_layer(d_model = self.d_model,
                                                                d_ffn = self.d_ffn,
                                                                dropout = self.dropout,
                                                                n_heads = self.n_head,
                                                                n_points = self.n_points)
        
        self.encoder = MSDeform_attn_transformer_encoder(encoder_layer=encoder_layer, n_encoder_layers=self.n_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(3, self.d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def forward(self, inputs, pos_embeds):

        input_flatten = []
        level_shapes = []
        embed = []
        B = inputs[0].shape[0]
        for L, (input, pos_embed) in enumerate(zip(inputs, pos_embeds)):
            _, _, H, W = input.shape
            
            level_shape = (H, W)
            level_shapes.append(level_shape)
            
            #(B, C, H, W) -> (B, HxW, C) input_1D 형태로 바꾸기
            input = input.flatten(2).transpose(1,2)
            input_flatten.append(input)
            
            pos_embed = pos_embed.flatten(2).transpose(1,2)
            #픽샐의 포지션 임배딩과 각 레벨의 임배딩 더하기
            level_pos_embed_added = pos_embed + self.level_embed[L].view(1, 1, -1)
            embed.append(level_pos_embed_added)
        #(B, T, C)
        input_flatten = torch.cat(input_flatten, 1)
        embed = torch.cat(embed, 1)
        level_shapes = torch.as_tensor(level_shapes, dtype=torch.long, device=input_flatten.device)
        #앞에 0 채우고 누적합 -> (0, ...)
        level_start_idx = torch.cat((level_shapes.new_zeros((1, )), level_shapes.prod(1).cumsum(0)[:-1]))

        memory = self.encoder(input=input_flatten, level_shapes=level_shapes, level_start_idx=level_start_idx, pos_embed=embed, B=B)

        return memory, level_shapes

class MSDeform_attention_pixel_decoder(nn.Module):
    def __init__(self, feature_channels ,mask_dim=256, d_model=256, n_head=8, n_encoder_layers=6, d_ffn=1024, dropout=0.1, n_points=4):
        super().__init__()
        
        self.feature_channels = feature_channels
        self.proj_dim =d_model

        #트랜스포머 모델
        self.transformer = MSDeform_attn_transformer_encoder_only(d_model, n_head, n_encoder_layers, d_ffn, dropout, n_points)
        #포지션 임배딩 모델
        self.pos_embed_layer = pixel_position_embedding(d_model)
        # 1/4를 마스크 차원 크기로 바꾸기
        self.mask_layer = nn.Conv2d(d_model, mask_dim, kernel_size=1, stride=1, padding=0)
        weight_init.c2_xavier_fill(self.mask_layer)

        set_channel = nn.Conv2d(in_channels=self.feature_channels[0], out_channels=d_model, kernel_size=1, stride=1, padding=0)
        weight_init.c2_xavier_fill(set_channel)
        self.set_channel = set_channel
        
        temp_conv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        weight_init.c2_xavier_fill(temp_conv)
        out_conv = nn.Sequential(
                                temp_conv,
                                nn.ReLU()
                                )  
        self.out_conv = out_conv
        
        self.input_proj = nn.ModuleList(self.make_input_proj_list())
        
        self.transformer_map = ["res5", "res4", "res3"]

    def make_input_proj_list(self):
        conv_list = []
        for i in [3, 2, 1]:
            conv_list.append(nn.Sequential(nn.Conv2d(self.feature_channels[i], self.proj_dim, kernel_size=1), nn.GroupNorm(32, self.proj_dim)))
        return conv_list
    
    def forward_features(self, features: dict):
        inputs = []
        pos_embeds = []
        for i, f in enumerate(self.transformer_map):
            projected = self.input_proj[i](features[f]).float()
            inputs.append(projected)
            pos_embeds.append(self.pos_embed_layer(projected))

        flatten_output, level_shapes = self.transformer(inputs, pos_embeds)
        n_of_each = level_shapes.prod(1).tolist()
        
        flatten_output = torch.split(flatten_output, n_of_each, dim=1)

        #원래 형태로 바꾸기
        B = flatten_output[0].shape[0]
        final_out = []
        for i, f in enumerate(flatten_output):
            final_out.append(f.transpose(1,2).view(B, -1, level_shapes[i][0], level_shapes[i][1]))

        res2 = features["res2"].float()
        channel_conv = self.set_channel(res2)
        res2_out = channel_conv + F.interpolate(final_out[-1], size=channel_conv.shape[-2:], mode='bilinear', align_corners=False)
        res2_out = self.out_conv(res2_out)
        res2_out = self.mask_layer(res2_out)
        final_out.append(res2_out)

        return final_out


