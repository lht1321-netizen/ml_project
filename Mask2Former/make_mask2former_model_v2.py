from torch import nn

from .backbone.backbone import D2SwinTransformer
from .pixel_decoder.pixel_decoder_v2 import MSDeform_attention_pixel_decoder
from .transformer_decoder.transformer_decoder import Transformer_Decoder

class Mask2Former(nn.Module):
    """
    Backbone (Swin) + Pixel Decoder (TEncoderPixelDecoder) + Transformer Decoder (Transformer_Decoder)
    를 조립하는 최종 Mask2Former 모델입니다.
    """

    # 이 클래스는 from_config를 사용하지 않고,
    # 필요한 설정값(cfg)을 직접 받아 하위 모듈을 초기화합니다.
    def __init__(self, backbone_cfg, pixel_decoder_cfg, transformer_decoder_cfg, input_shape):
        super().__init__()

        # --- 1. Backbone (Swin) 초기화 ---
        # input_shape은 (C, H, W)를 가정 (e.g., ShapeSpec(channels=3))
        # D2SwinTransformer.from_config는 cfg와 input_shape을 받음
        self.backbone = D2SwinTransformer(
            **D2SwinTransformer.from_config(backbone_cfg, input_shape)
        )

        # 백본의 출력 shape을 가져옴 (e.g., {"res2": ShapeSpec, ...})
        backbone_output_shape = self.backbone.output_shape()
        feature_channels = [backbone_output_shape[res].channels for res in ["res2", "res3", "res4", "res5"] ]

        conv_dim = pixel_decoder_cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = pixel_decoder_cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        decoder_channel_dim_list = [conv_dim, conv_dim, conv_dim, mask_dim]
        
        #msderomattn으로 변경
        self.pixel_decoder = MSDeform_attention_pixel_decoder(feature_channels=feature_channels,
                                                              mask_dim=mask_dim,
                                                              d_model=conv_dim)

        decoder_C_dim = pixel_decoder_cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM

        decoder_mask_dim = pixel_decoder_cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        self.transformer_decoder = Transformer_Decoder(
            C_dim=decoder_C_dim,
            n_head=transformer_decoder_cfg.MODEL.MASK_FORMER.NHEADS,
            dim_feedforward=transformer_decoder_cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            channel_dim_list=decoder_channel_dim_list,
            mask_dim=decoder_mask_dim,
            L=transformer_decoder_cfg.MODEL.MASK_FORMER.DEC_LAYERS, # (cfg에 이 항목이 필요합니다)
            dropout=transformer_decoder_cfg.MODEL.MASK_FORMER.DROPOUT,
            n_query=transformer_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES, # (cfg에 이 항목이 필요합니다)
            n_class=transformer_decoder_cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES # (cfg에 이 항목이 필요합니다)
        )

    def forward(self, x):
        # 1. 백본 실행 (Swin)
        # (Input: [B, 3, H, W])
        # (Output: {"res2": [B,C2,H/4,W/4], "res3": [B,C3,H/8,W/8], ...})
        features = self.backbone(x)

        # 2. 픽셀 디코더 실행 (TransformerEncoderPixelDecoder)
        feature_pyramid = self.pixel_decoder.forward_features(features)

        # (Input: [List of 4 Tensors])
        # (Output: {'predicted_class': ..., 'predicted_mask': ..., 'auxiliary_out': ...})
        outputs = self.transformer_decoder(feature_pyramid)

        return outputs