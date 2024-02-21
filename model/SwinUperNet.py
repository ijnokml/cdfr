import torch.nn as nn

from .fcn_head import FCNHead
from .swin_transformer import SwinTransformer
from .main_head import MainHead

from mmseg.ops import resize
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.05)
backbone_norm_cfg = dict(type='LN', requires_grad=True)


class SwinUperNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = SwinTransformer(
            pretrain_img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            # num_classes = self.num_classes,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT
        )
        self.decode_head = MainHead(
            in_channels=[96, 192, 384, 768],  # here
            num_classes=config.MODEL.NUM_CLASSES,
            in_index=[0, 1, 2, 3],
            channels=512,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        )
        self.auxiliary_head = FCNHead(
            in_channels=384,
            in_index=2, #超参数
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=config.MODEL.NUM_CLASSES,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
        )

    def forward(self, input_x):
        x = self.backbone(input_x)
        decode_outs = self.decode_head(x)
        aux_outs = self.auxiliary_head(x)
        # decode_outs = resize(
        #     input=decode_outs,
        #     size=input_x.shape[2:],
        #     mode='bilinear',
        #     align_corners=False)
        aux_outs = resize(
            input=aux_outs,
            size=input_x.shape[2:],
            mode='bilinear',
            align_corners=False)
        return decode_outs, aux_outs  # 主分类器，辅助分类器
