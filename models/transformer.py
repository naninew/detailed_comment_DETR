# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


# =============================================================================
# TRANSFORMER - KIẾN TRÚC CORE CỦA DETR
# =============================================================================
# Đây là thành phần trung tâm của DETR, chịu trách nhiệm xử lý thông tin giữa
# encoder (để hiểu hình ảnh) và decoder (để dự đoán objects). Kiến trúc này
# dựa trên transformer gốc nhưng có các điều chỉnh quan trọng:
#
# 1. Positional encodings được đưa vào multi-head attention thay vì cộng vào input
# 2. LayerNorm cuối cùng của encoder bị loại bỏ để đơn giản hóa kiến trúc
# 3. Decoder trả về stack của tất cả layers (cho auxiliary losses) thay vì chỉ layer cuối
#
# Tại sao dùng Transformer cho object detection:
# - Self-attention cho phép model capture global dependencies giữa tất cả regions
# - Encoder-decoder architecture phù hợp cho việc map từ image features sang object queries
# - Parallel processing của transformer nhanh hơn RNN/CNN trong training
# - Set-based prediction phù hợp với bản chất của detection (tập hợp objects)
# =============================================================================

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        # Khởi tạo encoder layer với cấu hình tiêu chuẩn của transformer
        # Mỗi encoder layer gồm: self-attention + feed-forward network với residual connections
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # Khởi tạo decoder layer với cross-attention để kết hợp information từ encoder
        # Mỗi decoder layer gồm: self-attention + cross-attention + feed-forward network
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    # ==========================================================================
    # XAVIER UNIFORM INITIALIZATION CHO CÁC WEIGHTS 2D
    # ==========================================================================
    # Phương thức này khởi tạo tất cả các weight matrices có số chiều > 1
    # sử dụng Xavier uniform initialization. Đây là kỹ thuật quan trọng giúp:
    # - Duy trì variance ổn định qua các layers trong quá trình forward/backward pass
    # - Tránh vanishing/exploding gradients đặc biệt quan trọng với deep transformers
    # - Giúp hội tụ nhanh hơn trong training
    # ==========================================================================
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ==========================================================================
    # FORWARD PASS - XỬ LÝ ẢNH QUA ENCODER VÀ DECODER
    # ==========================================================================
    # Đầu vào:
    # - src: Feature maps từ backbone [batch, channels, height, width]
    # - mask: Padding mask cho các vùng không hợp lệ
    # - query_embed: Learnable object queries [num_queries, hidden_dim]
    # - pos_embed: Positional encodings cho spatial information
    #
    # Quy trình xử lý:
    # 1. Flatten và permute feature maps từ NxCxHxW thành HWxNxC (chuẩn cho transformer)
    # 2. Tạo target queries ban đầu bằng zeros (sẽ được cộng với query_pos trong decoder)
    # 3. Encoder xử lý image features với self-attention và positional encoding
    # 4. Decoder thực hiện cross-attention giữa queries và encoder memory
    #
    # Đầu ra:
    # - hs: Stack of decoder outputs [num_layers, batch, num_queries, hidden_dim]
    # - memory: Encoder output reshaped lại thành dạng ảnh [batch, channels, height, width]
    # ==========================================================================
    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


# =============================================================================
# TRANSFORMER ENCODER - TRÍCH XUẤT ĐẶC TRƯNG HÌNH ẢNH TOÀN CỤC
# =============================================================================
# Encoder gồm nhiều identical encoder layers xếp chồng lên nhau.
# Chức năng chính:
# - Áp dụng self-attention để mỗi vị trí trong feature map có thể attend đến tất cả vị trí khác
# - Tích hợp positional encoding để giữ thông tin không gian
# - Tạo ra "memory" representation mà decoder sẽ attend vào
#
# Số lượng layers mặc định là 6, đủ sâu để capture complex patterns nhưng
# không quá sâu gây khó khăn cho optimization
# =============================================================================
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    # ==========================================================================
    # ENCODER FORWARD - ÁP DỤNG TUẦN TỰ CÁC ENCODER LAYERS
    # ==========================================================================
    # Phương thức này thực hiện forward pass qua tất cả encoder layers.
    # Mỗi layer nhận vào:
    # - src: Input features với shape [HW, batch, hidden_dim]
    # - pos: Positional encodings được cộng vào queries và keys trong self-attention
    # - mask: Key padding mask để ignore các padded positions
    #
    # Output của layer trước là input của layer sau, tạo thành deep network
    # LayerNorm cuối cùng (nếu có) được áp dụng sau tất cả layers
    # ==========================================================================
    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


# =============================================================================
# TRANSFORMER DECODER - GIẢI MÃ OBJECT QUERIES THÀNH DỰ ĐOÁN
# =============================================================================
# Decoder gồm nhiều identical decoder layers, mỗi layer thực hiện 3 bước:
# 1. Self-attention giữa các object queries (với query positional encoding)
# 2. Cross-attention giữa queries và encoder memory (image features)
# 3. Feed-forward network để transform features
#
# Đặc điểm quan trọng:
# - return_intermediate=True cho phép lấy outputs từ tất cả layers
#   để tính auxiliary losses, giúp gradient flow tốt hơn trong training sâu
# - Object queries ban đầu là zeros, học được thông tin qua query_pos embeddings
# =============================================================================
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    # ==========================================================================
    # DECODER FORWARD - XỬ LÝ OBJECT QUERIES QUA NHIỀU LAYERS
    # ==========================================================================
    # Phương thức này thực hiện iterative refinement của object queries:
    # - tgt: Target queries ban đầu (zeros) sẽ được cập nhật qua mỗi layer
    # - memory: Encoder output chứa image features đã được encode
    # - query_pos: Positional encodings cho queries, cộng vào trước attention
    # - pos: Positional encodings cho memory, dùng trong cross-attention
    #
    # Khi return_intermediate=True:
    # - Lưu lại output sau mỗi layer (đã qua normalization)
    # - Cho phép tính auxiliary losses ở nhiều depths khác nhau
    # - Giúp giải quyết vấn đề vanishing gradients trong deep networks
    # - Cải thiện chất lượng predictions, đặc biệt ở những layers đầu
    # ==========================================================================
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


# =============================================================================
# TRANSFORMER ENCODER LAYER - ĐƠN VỊ CƠ BẢN CỦA ENCODER
# =============================================================================
# Mỗi encoder layer bao gồm hai sub-layers chính:
# 1. Multi-head self-attention: Cho phép mỗi position attend đến tất cả positions khác
# 2. Feed-forward network: MLP 2 layers với ReLU/GELU activation
#
# Kiến trúc sử dụng:
# - Residual connections quanh mỗi sub-layer để giúp gradient flow
# - Layer normalization để ổn định training
# - Dropout để regularization
#
# Hai chế độ normalization:
# - normalize_before=False (post-norm): Norm sau residual connection (truyền thống)
# - normalize_before=True (pre-norm): Norm trước khi vào sub-layer (ổn định hơn)
# =============================================================================
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Multi-head self-attention cho phép mỗi spatial position attend đến tất cả positions khác
        # Số heads (nhead) chia nhỏ dimension để học nhiều aspects khác nhau của data
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # FFN gồm 2 linear layers với expansion ratio 4 (512 -> 2048 -> 512)
        # Expansion này giúp model có capacity đủ để learn complex transformations
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization để ổn định training và giúp gradient flow tốt hơn
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout regularization cho cả attention và FFN sub-layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    # ==========================================================================
    # HELPER FUNCTION - CỘNG POSITIONAL EMBEDDING VÀO TENSOR
    # ==========================================================================
    # Hàm utility đơn giản nhưng quan trọng: chỉ cộng pos vào tensor nếu pos tồn tại
    # Được sử dụng trong cả encoder và decoder để inject spatial information
    # Vào queries và keys trước khi tính attention (không cộng vào values)
    # ==========================================================================
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # ==========================================================================
    # POST-NORM FORWARD - KIẾN TRÚC NORM SAU RESIDUAL CONNECTION
    # ==========================================================================
    # Đây là kiến trúc transformer gốc từ "Attention is All You Need":
    # 1. Self-attention: q=k=(src+pos), v=src -> residual connection -> LayerNorm
    # 2. FFN: Linear1 -> Activation -> Dropout -> Linear2 -> residual -> LayerNorm
    #
    # Ưu điểm của post-norm:
    # - Đơn giản, phù hợp với các mô hình không quá sâu
    # - Được sử dụng rộng rãi trong các implementation ban đầu
    #
    # Nhược điểm:
    # - Có thể gặp vấn đề vanishing gradients với very deep networks
    # - Khó train hơn pre-norm architecture
    # ==========================================================================
    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # Bước 1: Self-attention với positional encoding
        # Queries và Keys đều được cộng với pos để inject spatial information
        # Values giữ nguyên để preserve original feature content
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # Residual connection giúp gradient flow trực tiếp qua skip connection
        src = src + self.dropout1(src2)
        # Layer normalization sau residual connection
        src = self.norm1(src)
        # Bước 2: Feed-forward network với expansion và compression
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    # ==========================================================================
    # PRE-NORM FORWARD - KIẾN TRÚC NORM TRƯỚC SUB-LAYER (KHUYẾN NGHỊ)
    # ==========================================================================
    # Kiến trúc pre-norm được đề xuất trong các nghiên cứu gần đây:
    # 1. LayerNorm -> Self-attention -> residual connection
    # 2. LayerNorm -> FFN -> residual connection
    #
    # Ưu điểm vượt trội của pre-norm:
    # - Gradient flow tốt hơn do normalized inputs vào mỗi sub-layer
    # - Ổn định hơn khi training với learning rate cao
    # - Hội tụ nhanh hơn và ít nhạy cảm với initialization
    # - Đặc biệt hiệu quả cho deep transformers (6+ layers)
    # ==========================================================================
    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    # ==========================================================================
    # FORWARD - LỰA CHỌN KIẾN TRÚC PRE-NORM HOẶC POST-NORM
    # ==========================================================================
    # Phương thức này dispatch đến forward_pre hoặc forward_post dựa trên flag
    # normalize_before. Đây là design pattern linh hoạt cho phép experiment
    # với cả hai kiến trúc mà không cần thay đổi code bên ngoài
    # ==========================================================================
    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


# =============================================================================
# TRANSFORMER DECODER LAYER - ĐƠN VỊ CƠ BẢN CỦA DECODER
# =============================================================================
# Decoder layer phức tạp hơn encoder layer với 3 sub-layers:
# 1. Masked self-attention: Các queries only attend đến chính nó (cho set prediction)
# 2. Cross-attention: Queries attend vào encoder memory (image features)
# 3. Feed-forward network: Transform combined features
#
# Khác biệt quan trọng so với encoder:
# - Có thêm cross-attention layer để kết hợp information từ encoder
# - Sử dụng query_pos riêng cho queries và pos cho memory
# - Ba norm layers tương ứng với ba sub-layers
# =============================================================================
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Self-attention cho object queries - chỉ allow queries attend đến nhau
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Cross-attention (multi-head attention) - queries attend vào encoder memory
        # Đây là cầu nối quan trọng giữa decoder và encoder, cho phép queries
        # extract relevant information từ image features
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Ba layer norms tương ứng với ba sub-layers: self-attn, cross-attn, FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Ba dropout layers cho regularization ở mỗi sub-layer
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    # ==========================================================================
    # HELPER FUNCTION - CỘNG POSITIONAL EMBEDDING VÀO TENSOR
    # ==========================================================================
    # Giống như trong encoder layer, nhưng ở đây dùng cho cả query_pos và pos
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # ==========================================================================
    # POST-NORM DECODER FORWARD - BA BƯỚC XỬ LÝ TRONG DECODER LAYER
    # ==========================================================================
    # Bước 1 - Self-Attention trên Queries:
    #   - q = k = tgt + query_pos (positional encoding cho queries)
    #   - v = tgt (original target features)
    #   - Cho phép các object queries trao đổi thông tin với nhau
    #
    # Bước 2 - Cross-Attention với Encoder Memory:
    #   - query = tgt + query_pos (queries đã được refine từ bước 1)
    #   - key = value = memory + pos (encoder output với positional encoding)
    #   - Đây là bước then chốt giúp queries "hỏi" encoder về image content
    #
    # Bước 3 - Feed-Forward Network:
    #   - Transform combined features qua MLP 2 layers
    #   - Cung cấp non-linearity và capacity cho model
    # ==========================================================================
    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # Bước 1: Self-attention giữa các object queries
        # Query positional encoding giúp phân biệt các queries khác nhau
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # Bước 2: Cross-attention giữa queries và encoder memory
        # Queries (với query_pos) attend vào memory (với pos) để extract information
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # Bước 3: Feed-forward network để transform features
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    # ==========================================================================
    # PRE-NORM DECODER FORWARD - KIẾN TRÚC NORM TRƯỚC SUB-LAYER
    # ==========================================================================
    # Tương tự như encoder, pre-norm trong decoder cũng mang lại lợi ích:
    # 1. Norm -> Self-attention -> residual
    # 2. Norm -> Cross-attention -> residual  
    # 3. Norm -> FFN -> residual
    #
    # Lưu ý: Trong pre-norm, normalized tensor được dùng cho cả computation
    # và làm output của mỗi sub-layer, khác với post-norm chỉ norm ở cuối
    # ==========================================================================
    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    # ==========================================================================
    # FORWARD - DISPATCH ĐẾN PRE-NORM HOẶC POST-NORM
    # ==========================================================================
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


# =============================================================================
# HELPER FUNCTION - TẠO N COPIES CỦA MỘT MODULE
# =============================================================================
# Hàm utility sử dụng copy.deepcopy để tạo N bản sao độc lập của một module.
# Được dùng để tạo các encoder/decoder layers giống hệt nhau về kiến trúc
# nhưng có parameters riêng biệt (không share weights).
#
# Tại sao dùng deepcopy thay vì tái sử dụng cùng một module:
# - Mỗi layer cần học các transformations khác nhau
# - Share weights sẽ giới hạn capacity của model
# - Deepcopy đảm bảo mỗi layer có独立的 parameters và gradients
# =============================================================================
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# =============================================================================
# BUILD TRANSFORMER FACTORY - HÀM KHỞI TẠO TRANSFORMER TỪ ARGS
# =============================================================================
# Factory function tạo transformer object từ command line arguments.
# Các hyperparameters quan trọng:
# - d_model (hidden_dim): Dimension của embeddings, mặc định 256 trong DETR
# - nhead: Số attention heads, mặc định 8
# - enc_layers/dec_layers: Số lượng encoder/decoder layers, mặc định 6
# - dim_feedforward: Hidden dimension của FFN, mặc định 2048 (4x d_model)
# - dropout: Dropout rate cho regularization
# - pre_norm: Sử dụng pre-norm architecture nếu True
# - return_intermediate_dec: Luôn True để hỗ trợ auxiliary losses
# =============================================================================
def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


# =============================================================================
# ACTIVATION FUNCTION FACTORY - LỰA CHỌN HÀM KÍCH HOẠT
# =============================================================================
# Hỗ trợ ba activation functions phổ biến:
# - ReLU: Rectified Linear Unit, mặc định và được sử dụng rộng rãi nhất
# - GELU: Gaussian Error Linear Unit, mượt mà hơn ReLU, dùng trong BERT
# - GLU: Gated Linear Unit, có cơ chế gating đặc biệt
#
# Việc lựa chọn activation function ảnh hưởng đến:
# - Khả năng biểu diễn non-linear của model
# - Gradient flow trong backpropagation
# - Tốc độ hội tụ và chất lượng final model
# =============================================================================
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
