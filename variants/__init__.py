"""Variant registry for PC network architectures."""

from config import (
    VARIANT_BASELINE, VARIANT_RESNET, VARIANT_BF,
    VARIANT_BF_V2, VARIANT_DYT, VARIANT_DYT_V2,
    VARIANT_DYT_V3, VARIANT_MUPC, VARIANT_REC_LRA,
    VARIANT_CNN_REC_LRA, VARIANT_RES_ERROR_NET,
    VARIANT_RES_ERROR_NET_RESNET18,
)
from variants.baseline import BaselineVariant
from variants.resnet import ResNetVariant
from variants.batchfreezing import BatchFreezingVariant
from variants.batchfreezing_v2 import BatchFreezingV2Variant
from variants.dyt import DyTVariant
from variants.dyt_v2 import DyTV2Variant
from variants.dyt_v3 import DyTV3Variant
from variants.mupc import MuPCVariant
from variants.rec_lra import RecLRAVariant
from variants.cnn_rec_lra import CNNRecLRAVariant
from variants.res_error_net import ResErrorNetVariant
from variants.res_error_net_resnet18 import ResErrorNetResNet18Variant

VARIANT_REGISTRY = {
    VARIANT_BASELINE: BaselineVariant,
    VARIANT_RESNET: ResNetVariant,
    VARIANT_BF: BatchFreezingVariant,
    VARIANT_BF_V2: BatchFreezingV2Variant,
    VARIANT_DYT: DyTVariant,
    VARIANT_DYT_V2: DyTV2Variant,
    VARIANT_DYT_V3: DyTV3Variant,
    VARIANT_MUPC: MuPCVariant,
    VARIANT_REC_LRA: RecLRAVariant,
    VARIANT_CNN_REC_LRA: CNNRecLRAVariant,
    VARIANT_RES_ERROR_NET: ResErrorNetVariant,
    VARIANT_RES_ERROR_NET_RESNET18: ResErrorNetResNet18Variant,
}


def get_variant(name):
    """Instantiate a variant by name."""
    cls = VARIANT_REGISTRY[name]
    return cls()
