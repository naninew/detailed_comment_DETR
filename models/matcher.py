# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


# =============================================================================
# HUNGARIAN MATCHER - BỘ GHÉP CẶP HUNGARY CHO DETR
# =============================================================================
# Đây là thành phần then chốt trong kiến trúc DETR, thực hiện việc ghép cặp 1-1
# giữa các dự đoán (predictions) và ground truth targets. Khác với các phương pháp
# detection truyền thống sử dụng NMS (Non-Maximum Suppression), DETR áp dụng
# Hungarian algorithm để giải quyết bài toán Linear Sum Assignment Problem (LSAP),
# đảm bảo mỗi ground truth object chỉ được ghép với duy nhất một prediction.
# 
# Lý do sử dụng Hungarian Matcher:
# 1. Loại bỏ nhu cầu về anchor boxes và NMS post-processing
# 2. Đảm bảo tính duy nhất trong việc gán nhãn, tránh trường hợp multiple predictions
#    cho cùng một object
# 3. Tối ưu hóa toàn cục thay vì greedy matching như các phương pháp khác
# 4. Phù hợp với set-based prediction nature của transformer
# =============================================================================

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    # ==========================================================================
    # PHƯƠNG THỨC FORWARD - TÍNH TOÁN COST MATRIX VÀ ÁP DỤNG HUNGARIAN ALGORITHM
    # ==========================================================================
    # Phương thức này thực hiện hai bước chính:
    # 1. Xây dựng cost matrix C với kích thước [batch_size, num_queries, num_targets]
    #    trong đó mỗi phần tử C[i,j] biểu thị chi phí để ghép prediction i với target j
    # 2. Áp dụng linear_sum_assignment từ scipy để tìm optimal matching với tổng cost nhỏ nhất
    #
    # Cost matrix được tổ hợp từ 3 thành phần:
    # - cost_class: Classification cost dựa trên predicted class probability
    # - cost_bbox: L1 distance giữa predicted và ground truth box coordinates
    # - cost_giou: Generalized IoU cost để đo độ overlap giữa các boxes
    #
    # Lưu ý quan trọng: torch.no_grad() được sử dụng vì đây là bước matching,
    # không cần tính gradient cho quá trình này
    # ==========================================================================
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten toàn bộ batch và queries thành một ma trận lớn để tính toán song song
        # Kích thước: [batch_size * num_queries, num_classes] và [batch_size * num_queries, 4]
        # Việc flatten này cho phép tính vectorized operations thay vì loop qua từng sample
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Concatenate tất cả ground truth labels và boxes từ toàn bộ targets trong batch
        # Đây là bước chuẩn bị để tính cost matrix giữa tất cả predictions và tất cả ground truths
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # Cost class âm vì muốn tối đa hóa probability của đúng class (minimize -prob = maximize prob)
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        # Sử dụng cdist để tính khoảng cách L1 giữa mọi cặp predicted và target boxes
        # Khoảng cách L1 (absolute difference) ổn định hơn L2 trong trường hợp outliers
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        # Generalized IoU cung cấp thông tin về độ overlap ngay cả khi boxes không giao nhau
        # Dấu âm vì muốn minimize cost tương đương với maximize GIoU
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix - Tổ hợp tuyến tính của 3 cost components với weights đã chỉ định
        # Công thức: C = w_bbox * L1_distance + w_class * (1 - prob) + w_giou * (1 - GIoU)
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        # Tách cost matrix theo số lượng targets của mỗi sample trong batch
        # Sau đó áp dụng Hungarian algorithm độc lập cho từng sample
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# =============================================================================
# BUILD MATCHER FACTORY - HÀM KHỞI TẠO MATCHER TỪ ARGUMENTS
# =============================================================================
# Design pattern factory function giúp tách biệt việc khởi tạo đối tượng khỏi
# logic chính của model. Các arguments được truyền từ command line hoặc config file
# sẽ được sử dụng để cấu hình weights cho 3 thành phần cost trong matching
# =============================================================================
def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
