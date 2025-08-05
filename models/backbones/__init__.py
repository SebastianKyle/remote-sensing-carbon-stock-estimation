from .cross_attention_resnet50 import CAResNet
from .cbam_resnet50 import CBAMResNet
from .lka_resnet50 import LKAResNet
from .msca_resnet50 import MSCAResNet
from .aattn_resnet50 import AAttnResNet
from .swin import SwinTransformer
from .convnext import CBAMConvNeXt
from .hcf_backbones import HCFResNet, HCFRFResNet, HCFLKAResNet, HCFCAResNet, HCFEfficientNet, HCFSwinEfficientNet, HCFResNetNoFPN
from .coordinate_attention_resnet50 import CORResNet
from .caf_resnet50 import CAFResNet
from .transcba_resnet50 import TransCBAResNet

backbones = {
    'ca_resnet50': CAResNet,
    'cbam_resnet50': CBAMResNet,
    'lka_resnet50': LKAResNet,
    'msca_resnet50': MSCAResNet,
    'swin_transformer': SwinTransformer,
    'aattn_resnet50': AAttnResNet,
    'cbam_convnext': CBAMConvNeXt,
    'hcf_resnet50': HCFResNet,
    'hcf_resnet50_nofpn': HCFResNetNoFPN,
    'hcf_efficientnet': HCFEfficientNet,
    'hcf_swin_efficientnet': HCFSwinEfficientNet,
    'hcf_rf_resnet50': HCFRFResNet,
    'hcf_lka_resnet50': HCFLKAResNet,
    'hcf_ca_resnet50': HCFCAResNet,
    'cor_resnet50': CORResNet,
    'caf_resnet50': CAFResNet,
    'transcba_resnet50': TransCBAResNet,
}

__all__ = ['CBAMResNet', 'LKAResNet', 'MSCAResNet', 'CAResNet', 'SwinTransformer', 'backbones', 'AAttnResNet', 'CBAMConvNeXt', 'HCFResNet', 'HCFEfficientNet', 'HCFSwinEfficientnet', 'HCFRFResNet', 'HCFLKAResNet', 'CORResNet', 'CAFResNet', 'HCFCAResNet', 'TransCBAResNet']
