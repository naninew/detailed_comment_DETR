# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


# =============================================================================
# CLASS FROZENBATCHNORM2D: BATCH NORM KHÔNG TRAINABLE
# =============================================================================
# BatchNorm2d với statistics và affine parameters được cố định (không update gradient).
# Lý do sử dụng trong DETR:
# 1. Ổn định training khi fine-tuning backbone pretrained
# 2. Giảm memory consumption và computation cost
# 3. Tránh hiện tượng catastrophic forgetting khi backbone đã được pretrain tốt
#
# Implementation khác biệt so với torchvision: thêm eps trước rsqrt để tránh NaN
# với các models không phải ResNet-18/34/50/101.
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # ---------------------------------------------------------------------
        # CONTROL GRADIENT FLOW: Chỉ train một số layers nhất định của backbone
        # ---------------------------------------------------------------------
        # Chiến lược fine-tuning selective:
        # - Nếu lr_backbone > 0: train các layers cao (layer2,3,4) chứa semantic information
        # - Các layers thấp (layer1, conv1) giữ frozen vì học low-level features có thể reuse
        # Cách tiếp cận này cân bằng giữa adaptation và preservation of pretrained knowledge.
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # ---------------------------------------------------------------------
        # INTERMEDIATE LAYERS EXTRACTION: Multi-scale features cho segmentation
        # ---------------------------------------------------------------------
        # return_interm_layers=True khi cần masks (segmentation task).
        # IntermediateLayerGetter tự động extract features từ nhiều stages của ResNet:
        # - layer1: resolution cao, chi tiết fine-grained (edges, textures)
        # - layer2,3: mid-level features
        # - layer4: resolution thấp, semantic information mạnh
        # Việc sử dụng multi-scale features critical cho panoptic/instance segmentation.
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # -----------------------------------------------------------------
            # MASK INTERPOLATION: Đồng bộ mask với feature map resolution
            # -----------------------------------------------------------------
            # Khi backbone downsamples ảnh (thường 32x), mask cũng phải được 
            # interpolate tương ứng để maintain spatial correspondence.
            # Mask này sau đó được dùng trong transformer attention để ignore padding.
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


# =============================================================================
# CLASS BACKBONE: RESNET BACKBONE VỚI FROZEN BATCH NORM
# =============================================================================
# Wrapper quanh torchvision ResNet models với hai customization:
# 1. FrozenBatchNorm2d thay vì BatchNorm2d thường
# 2. Hỗ trợ dilation ở layer4 cho dense prediction tasks
#
# Dilation (atrous convolution): Giữ nguyên resolution feature maps trong khi
# tăng receptive field, quan trọng cho detection/segmentation objects nhỏ.
class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        # num_channels là số channels của output feature map cuối cùng
        # ResNet-18/34: 512 channels (BasicBlock)
        # ResNet-50/101: 2048 channels (Bottleneck)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


# =============================================================================
# CLASS JOINLER: KẾT HỢP BACKBONE + POSITION ENCODING
# =============================================================================
# Sequential module đóng gói cả backbone và position encoding thành một interface
# thống nhất. Design pattern này:
# 1. Đơn giản hóa forward pass trong DETR model
# 2. Đảm bảo positional encoding luôn được compute đồng bộ với features
# 3. Trả về list của NestedTensor và pos encodings cho từng scale
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            # -----------------------------------------------------------------
            # POSITIONAL ENCODING GENERATION
            # -----------------------------------------------------------------
            # Transformer không có inherent notion về spatial positions như CNN.
            # Positional encodings inject spatial information vào features thông qua
            # additive embeddings. Mỗi pixel location có một unique encoding vector.
            # Xem position_encoding.py để hiểu chi tiết về sine vs learned embeddings.
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
