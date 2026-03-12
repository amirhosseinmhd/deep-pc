"""Variant registry for PC network architectures."""

from config import (
    VARIANT_BASELINE, VARIANT_RESNET, VARIANT_BF,
    VARIANT_BF_V2, VARIANT_DYT, VARIANT_DYT_V2,
    VARIANT_MUPC,
)
from variants.baseline import BaselineVariant
from variants.resnet import ResNetVariant
from variants.batchfreezing import BatchFreezingVariant
from variants.batchfreezing_v2 import BatchFreezingV2Variant
from variants.dyt import DyTVariant
from variants.dyt_v2 import DyTV2Variant
from variants.mupc import MuPCVariant

VARIANT_REGISTRY = {
    VARIANT_BASELINE: BaselineVariant,
    VARIANT_RESNET: ResNetVariant,
    VARIANT_BF: BatchFreezingVariant,
    VARIANT_BF_V2: BatchFreezingV2Variant,
    VARIANT_DYT: DyTVariant,
    VARIANT_DYT_V2: DyTV2Variant,
    VARIANT_MUPC: MuPCVariant,
}


def get_variant(name):
    """Instantiate a variant by name."""
    cls = VARIANT_REGISTRY[name]
    return cls()
