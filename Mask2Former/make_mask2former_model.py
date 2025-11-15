from torch import nn

from .backbone.backbone import D2SwinTransformer
from .pixel_decoder.pixel_decoder import TransformerEncoderPixelDecoder
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

        # --- 2. Pixel Decoder (FPN + Encoder) 초기화 ---
        # TransformerEncoderPixelDecoder.from_config는 cfg와 backbone_output_shape을 받음
        pixel_decoder_params = TransformerEncoderPixelDecoder.from_config(
            pixel_decoder_cfg, backbone_output_shape
        )
        self.pixel_decoder = TransformerEncoderPixelDecoder(**pixel_decoder_params)

        # --- 3. Transformer Decoder 초기화 ---
        # Transformer_Decoder는 cfg가 아닌 개별 인자를 받습니다.
        # (주의: channel_dim_list와 C_dim, mask_dim이 Pixel Decoder와 일치해야 함)

        # Pixel Decoder의 FPN 출력 채널 (conv_dim)
        # (res5 -> 1/32, res4 -> 1/16, res3 -> 1/8)
        # (주의: TransformerEncoderPixelDecoder의 multi_scale_features는 3개만 반환함)
        conv_dim = pixel_decoder_cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM

        # Pixel Decoder의 최종 마스크 특징 채널 (mask_dim)
        # (res2 -> 1/4)
        mask_dim = pixel_decoder_cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        # Transformer_Decoder에 전달할 채널 리스트
        # [1/32, 1/16, 1/8, 1/4] 스케일에 해당
        # (주의: Transformer_Decoder의 projection_modules[i]는 feature_pyramid[i]에 적용됨)
        # (주의: Transformer_Decoder의 forward는 3개의 feature map(0,1,2)과 1개의 mask map(3)을 사용함)
        # (주의: TransformerEncoderPixelDecoder의 multi_scale_features는 3개(conv_dim)를,
        #  mask_features는 1개(mask_dim)를 반환함. channel_dim_list와 순서/개수가 맞지 않음)

        # --- [충돌 해결] ---
        # 제공된 Transformer_Decoder의 forward는 4개의 feature_pyramid 입력을 받아
        # [0], [1], [2] (3개)는 cross-attention에, [3] (1개)는 마스크 예측에 사용합니다.

        # TransformerEncoderPixelDecoder의 multi_scale_features (3개)를 [0, 1, 2]에,
        # mask_features (1개)를 [3]에 매핑합니다.

        # (Pixel Decoder의 multi_scale_features는 [res5(1/32), res4(1/16), res3(1/8)] 순서)
        # (Pixel Decoder의 mask_features는 [res2(1/4)] 스케일)

        # channel_dim_list (Transformer_Decoder가 기대하는 입력 채널)
        # [1/32, 1/16, 1/8, 1/4]
        # (res5, res4, res3의 채널은 pixel_decoder의 conv_dim)
        # (res2의 채널은 pixel_decoder의 mask_dim)
        decoder_channel_dim_list = [conv_dim, conv_dim, conv_dim, mask_dim]

        # Transformer_Decoder의 C_dim (내부 Hidden Dim)
        # pixel_decoder_cfg의 conv_dim과 동일하게 맞추는 것이 일반적입니다.
        decoder_C_dim = pixel_decoder_cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM

        # Transformer_Decoder의 mask_dim (MLP 출력 Dim)
        # pixel_decoder_cfg의 mask_dim과 동일해야 합니다.
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
        # (Input: features dict)
        # (Output: mask_features [B, mask_dim, H/4, W/4],
        #          transformer_encoder_features [B, conv_dim, H/32, W/32],
        #          multi_scale_features [List of 3 Tensors: 1/32, 1/16, 1/8 scale])
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(features)

        # 3. 트랜스포머 디코더 실행 (Transformer_Decoder)
        # Transformer_Decoder는 [1/32, 1/16, 1/8, 1/4] 스케일의 리스트를 기대합니다.

        # (Pixel Decoder의 multi_scale_features는 [res5(1/32), res4(1/16), res3(1/8)])
        # (Pixel Decoder의 mask_features는 res2(1/4) 스케일)

        # (주의) TransformerEncoderPixelDecoder의 multi_scale_features는
        # [res5, res4, res3] (High-res to Low-res가 아님)
        # 원본 BasePixelDecoder의 순서는 low-to-high ([res5, res4, res3]) 입니다.
        # Transformer_Decoder의 forward는 [1/32, 1/16, 1/8] 순서를 기대합니다.

        # feature_pyramid 리스트 생성: [feat_1/32, feat_1/16, feat_1/8, feat_1/4]
        feature_pyramid = [
            multi_scale_features[0], # 1/32 (res5)
            multi_scale_features[1], # 1/16 (res4)
            multi_scale_features[2], # 1/8 (res3)
            mask_features            # 1/4 (res2, mask_dim 채널)
        ]

        # (Input: [List of 4 Tensors])
        # (Output: {'predicted_class': ..., 'predicted_mask': ..., 'auxiliary_out': ...})
        outputs = self.transformer_decoder(feature_pyramid)

        return outputs