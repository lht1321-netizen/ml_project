# SimpleNamespace를 사용하기 위해 import
import types
import torch
from fvcore.common.config import CfgNode as _CfgNode
from .helper.helper import ShapeSpec
from .make_mask2former_model import Mask2Former

# 1.1: 백본(Swin) 설정
# (backbone코드.ipynb의 D2SwinTransformer.from_config가 참조할 값들)
backbone_cfg = _CfgNode() # fvcore CfgNode 사용
backbone_cfg.MODEL = _CfgNode()
backbone_cfg.MODEL.SWIN = _CfgNode()
backbone_cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
backbone_cfg.MODEL.SWIN.PATCH_SIZE = 4
backbone_cfg.MODEL.SWIN.EMBED_DIM = 96
backbone_cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
backbone_cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
backbone_cfg.MODEL.SWIN.WINDOW_SIZE = 7
backbone_cfg.MODEL.SWIN.MLP_RATIO = 4.0
backbone_cfg.MODEL.SWIN.QKV_BIAS = True
backbone_cfg.MODEL.SWIN.QK_SCALE = None
backbone_cfg.MODEL.SWIN.DROP_RATE = 0.0
backbone_cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
backbone_cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
backbone_cfg.MODEL.SWIN.APE = False
backbone_cfg.MODEL.SWIN.PATCH_NORM = True
backbone_cfg.MODEL.SWIN.USE_CHECKPOINT = False
backbone_cfg.MODEL.SWIN.OUT_INDICES = (0, 1, 2, 3) # Transformer_Decoder가 4개 스케일을 가정

# 1.2: 픽셀 디코더 (TEncoderPixelDecoder) 설정
# (pixeldecoder_tem_fpn...ipynb의 TransformerEncoderPixelDecoder.from_config가 참조)
pixel_decoder_cfg = _CfgNode()
pixel_decoder_cfg.MODEL = _CfgNode()
pixel_decoder_cfg.MODEL.SEM_SEG_HEAD = _CfgNode()
pixel_decoder_cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
pixel_decoder_cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
pixel_decoder_cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
pixel_decoder_cfg.MODEL.SEM_SEG_HEAD.NORM = "GN"
# (TransformerEncoderPixelDecoder 내부의 인코더 설정)
pixel_decoder_cfg.MODEL.MASK_FORMER = _CfgNode()
pixel_decoder_cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
pixel_decoder_cfg.MODEL.MASK_FORMER.NHEADS = 8
pixel_decoder_cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 1024
pixel_decoder_cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 3
pixel_decoder_cfg.MODEL.MASK_FORMER.PRE_NORM = False

# 1.3: 트랜스포머 디코더 (Transformer_Decoder) 설정
# (TransformDecoder.py의 Transformer_Decoder가 참조)
transformer_decoder_cfg = _CfgNode()
transformer_decoder_cfg.MODEL = _CfgNode()
transformer_decoder_cfg.MODEL.MASK_FORMER = _CfgNode()
transformer_decoder_cfg.MODEL.MASK_FORMER.NHEADS = 8
transformer_decoder_cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048 # (TransformDecoder.py 기본값)
transformer_decoder_cfg.MODEL.MASK_FORMER.DEC_LAYERS = 3 # (TransformDecoder.py 기본값 L=3)
transformer_decoder_cfg.MODEL.MASK_FORMER.DROPOUT = 0.0 # (TransformDecoder.py 기본값)
transformer_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100 # (TransformDecoder.py 기본값)
transformer_decoder_cfg.MODEL.SEM_SEG_HEAD = _CfgNode()
transformer_decoder_cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1 # (TransformDecoder.py 기본값: 건물 1개 클래스)

# 2. 모델 초기화
print("🚀 모델 초기화를 시작합니다...")
# (주의: 전처리에서 1024x1024 타일을 사용했지만,
# 스모크 테스트는 메모리를 아끼기 위해 256x256으로 진행)
input_shape = ShapeSpec(channels=3, height=256, width=256)

model = Mask2Former(
    backbone_cfg,
    pixel_decoder_cfg,
    transformer_decoder_cfg,
    input_shape
)

# 3. 스모크 테스트 (Smoketest)
# 256x256 크기의 임의 이미지 1장 (배치 1)
dummy_input = torch.randn(1, 3, 256, 256)

print("\n✅ 모델 초기화 성공!")
print("--- 모델에 [1, 3, 256, 256] 텐서 입력 ---")

# 4. 모델 실행 (Forward pass)
model.eval() # 추론 모드로 설정
with torch.no_grad(): # 그래디언트 계산 안함
    outputs = model(dummy_input)

print("\n✅ Forward Pass 성공!")
print("--- 최종 출력(outputs) 형태 ---")
print(f"predicted_class: {outputs['predicted_class'].shape}")
print(f"predicted_mask: {outputs['predicted_mask'].shape}")
print(f"auxiliary_out (보조 출력 개수): {len(outputs['auxiliary_out'])}")
if len(outputs['auxiliary_out']) > 0:
    print(f"  -> (예: 보조출력 0번) class: {outputs['auxiliary_out'][0]['predicted_class'].shape}")
    print(f"  -> (예: 보조출력 0번) mask: {outputs['auxiliary_out'][0]['predicted_mask'].shape}")

print("\n🎉 모든 뼈대 코드가 성공적으로 조립되었습니다!")
     