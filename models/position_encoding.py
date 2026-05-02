# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


# =============================================================================
# POSITIONAL ENCODING - MÃ HÓA VỊ TRÍ CHO TRANSFORMER TRONG DETR
# =============================================================================
# Transformer architecture không có inherent notion về vị trí không gian như CNN.
# Do đó, positional encoding là thành phần bắt buộc để cung cấp thông tin về vị trí
# của các pixels trong feature maps. DETR sử dụng hai loại positional encoding:
#
# 1. Sine-based positional encoding (PositionEmbeddingSine):
#    - Sử dụng hàm sin/cos với các tần số khác nhau để mã hóa tọa độ
#    - Tương tự như trong "Attention is All You Need" nhưng mở rộng cho 2D images
#    - Ưu điểm: Không cần học thêm parameters, có thể generalize sang kích thước ảnh khác
#
# 2. Learned positional encoding (PositionEmbeddingLearned):
#    - Sử dụng learnable embeddings cho mỗi vị trí hàng và cột
#    - Ưu điểm: Có thể học được patterns đặc thù cho bài toán detection
#    - Nhược điểm: Cố định số lượng positions (50x50 trong implementation này)
#
# Cả hai phương pháp đều tạo ra positional embeddings có cùng dimension với 
# feature maps từ backbone (hidden_dim) để có thể cộng trực tiếp vào features
# =============================================================================

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    # ==========================================================================
    # FORWARD - TẠO SINE-BASED POSITIONAL EMBEDDINGS CHO FEATURE MAPS 2D
    # ==========================================================================
    # Phương thức này tạo positional embeddings sử dụng hàm sin và cos với các
    # tần số khác nhau, mở rộng ý tưởng từ transformer 1D sang ảnh 2D.
    #
    # Quy trình thực hiện:
    # 1. Tính cumulative sum của mask để xác định tọa độ y và x của mỗi pixel
    # 2. Chuẩn hóa tọa độ về khoảng [0, 2*pi] nếu normalize=True
    # 3. Tạo dim_t là dãy các lũy thừa của temperature cho việc chia tần số
    # 4. Áp dụng sin cho chiều chẵn và cos cho chiều lẻ của embedding
    # 5. Concatenate pos_x và pos_y để tạo final positional encoding
    #
    # Tại sao dùng sin/cos với nhiều tần số:
    # - Tần số cao (temperature^0) mã hóa thông tin vị trí chi tiết
    # - Tần số thấp (temperature^(2k/d)) mã hóa thông tin vị trí tổng quát
    # - Giúp model capture được cả local và global spatial relationships
    # ==========================================================================
    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# =============================================================================
# LEARNED POSITIONAL EMBEDDING - MÃ HÓA VỊ TRÍ HỌC ĐƯỢC
# =============================================================================
# Khác với sine encoding, learned embedding sử dụng các vectors có thể học được
# cho mỗi vị trí hàng và cột trong feature map. Cách tiếp cận này:
#
# 1. Tạo hai bảng embedding riêng biệt: row_embed (cho trục y) và col_embed (cho trục x)
# 2. Mỗi bảng có 50 entries, tương ứng với tối đa 50 positions theo mỗi chiều
# 3. Kết hợp row và column embeddings thông qua concatenation
# 4. Broadcast cho toàn bộ batch
#
# Ưu điểm của learned embedding:
# - Có thể học được đặc trưng vị trí tối ưu cho bài toán detection cụ thể
# - Linh hoạt hơn trong việc capture spatial patterns phức tạp
#
# Nhược điểm:
# - Số parameters tăng lên (50 * hidden_dim * 2)
# - Không generalize tốt sang ảnh có kích thước lớn hơn 50x50
# =============================================================================
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    # ==========================================================================
    # KHỞI TẠO PARAMETERS CHO LEARNED EMBEDDINGS
    # ==========================================================================
    # Sử dụng uniform initialization cho cả row và column embeddings
    # Việc khởi tạo đều đặn giúp đảm bảo các vị trí khác nhau có embeddings
    # phân biệt ngay từ đầu, tránh hiện tượng collapse trong quá trình training
    # ==========================================================================
    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    # ==========================================================================
    # FORWARD - TẠO LEARNED POSITIONAL EMBEDDINGS BẰNG CÁCH TRA CỨU TABLE
    # ==========================================================================
    # Quy trình:
    # 1. Xác định chiều cao h và rộng w của feature map đầu vào
    # 2. Tạo indices i (cho columns) và j (cho rows) từ 0 đến w-1 và h-1
    # 3. Tra cứu embeddings từ row_embed và col_embed tables
    # 4. Broadcast và concatenate để tạo grid 2D positional embeddings
    # 5. Permute và repeat cho toàn bộ batch
    #
    # Kết quả: pos tensor có shape [batch_size, hidden_dim, h, w]
    # Mỗi vị trí (y, x) trong feature map có một unique positional embedding
    # là concatenation của row_embed[y] và col_embed[x]
    # ==========================================================================
    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


# =============================================================================
# BUILD POSITION ENCODING FACTORY - HÀM KHỞI TẠO POSITIONAL ENCODING
# =============================================================================
# Factory function lựa chọn loại positional encoding dựa trên argument
# Cấu hình mặc định sử dụng sine encoding (v2/sine) vì tính generalize tốt
# Số lượng features được chia đôi (hidden_dim // 2) vì sẽ concatenate
# pos_x và pos_y, mỗi cái chiếm một nửa dimension
# =============================================================================
def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
