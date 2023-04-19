# --------------------------------------------------------
# Pytorch Yolov2
# Licensed under The MIT License [see LICENSE for details]
# Written by Jingru Tan
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from config import config as cfg
from util.bbox import generate_all_anchors, xywh2xxyy, box_transform_inv, box_ious, xxyy2xywh, box_transform
import torch.nn.functional as F


def build_target(output, gt_data, H, W):
    """
    Build the training target for output tensor

    Arguments:

    output_data -- tuple (delta_pred_batch, conf_pred_batch, class_pred_batch), output data of the yolo network
    gt_data -- tuple (gt_boxes_batch, gt_classes_batch, num_boxes_batch), ground truth data

    delta_pred_batch -- tensor of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
    conf_pred_batch -- tensor of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
    class_score_batch -- tensor of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2, ..)

    gt_boxes_batch -- tensor of shape (B, N, 4), ground truth boxes, normalized values
                       (x1, y1, x2, y2) range 0~1
    gt_classes_batch -- tensor of shape (B, N), ground truth classes (cls)
    num_obj_batch -- tensor of shape (B, 1). number of objects

    Returns:
    iou_target -- tensor of shape (B, H * W * num_anchors, 1)
    iou_mask -- tensor of shape (B, H * W * num_anchors, 1)
    box_target -- tensor of shape (B, H * W * num_anchors, 4)
    box_mask -- tensor of shape (B, H * W * num_anchors, 1)
    class_target -- tensor of shape (B, H * W * num_anchors, 1)
    class_mask -- tensor of shape (B, H * W * num_anchors, 1)
    """
    # [N, H*W*Num_anchors, 4]: sigmoid(t_x) / sigmoid(t_y) / e^t_w / e^t_h
    delta_pred_batch = output[0]
    # [N, H*W*Num_anchors, 1]: 预测框置信度
    conf_pred_batch = output[1]
    # [N, H*W*Num_anchors, Num_classes]: 分类输出
    class_score_batch = output[2]

    # [N, Num_objs, 4], Num_objs: 该批次数据中某一幅图像拥有的最多的标注框个数
    # 标注框坐标, 相对于图像大小
    gt_boxes_batch = gt_data[0]
    # [N, Num_objs]
    # 标注框对应类别下标
    gt_classes_batch = gt_data[1]
    # 每幅图像的标注框数目
    num_boxes_batch = gt_data[2]

    # [N] 批量大小
    bsize = delta_pred_batch.size(0)
    # [Num_anchors] 锚点框个数
    num_anchors = 5  # hard code for now

    # initial the output tensor
    # we use `tensor.new()` to make the created tensor has the same devices and data type as input tensor's
    # what tensor is used doesn't matter
    #
    # 这个应该是说置信度的
    # iou_target: [N, H*W, Num_anchors, 1]
    iou_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    # iou_mask: [N, H*W, Num_anchors, 1]
    # 置信度掩码, 包含了负责标注框的置信度预测(趋近于1)以及不包含目标的网格以及锚点框的置信度预测(趋近于0)
    iou_mask = delta_pred_batch.new_ones((bsize, H * W, num_anchors, 1)) * cfg.noobject_scale

    # target和mask, 看起来target用于损失计算的真值, mask用于判断哪些预测框用于损失计算
    # [N, H*W, Num_anchors, 4]
    box_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 4))
    # [N, H*W, Num_anchors, 1]
    box_mask = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

    # [N, H*W, Num_anchors, 1]
    class_target = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    # [N, H*W, Num_anchors, 1]
    class_mask = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

    # get all the anchors
    # [5, 2] 锚点框
    anchors = torch.FloatTensor(cfg.anchors)

    # note: the all anchors' xywh scale is normalized by the grid width and height, i.e. 13 x 13
    # this is very crucial because the predict output is normalized to 0~1, which is also
    # normalized by the grid width and height
    #
    # 生成网格锚点框
    # [H*W*Num_anchors, 4]
    all_grid_xywh = generate_all_anchors(anchors, H, W)  # shape: (H * W * num_anchors, 4), format: (x, y, w, h)
    # [H*W*Num_anchors, 4] -> [N, H*W*Num_anchors, 4]
    # 这一步的目的是为了保持相同的数据类型torch.dtype以及所处设备torch.device
    all_grid_xywh = delta_pred_batch.new(*all_grid_xywh.size()).copy_(all_grid_xywh)
    all_anchors_xywh = all_grid_xywh.clone()
    # [x1, y1, w, h] -> [xc, yc, w, h]
    all_anchors_xywh[:, 0:2] += 0.5
    if cfg.debug:
        print('all grid: ', all_grid_xywh[:12, :])
        print('all anchor: ', all_anchors_xywh[:12, :])
    # [xc, yc, w, h] -> [x1, y1, x2, y2]
    all_anchors_xxyy = xywh2xxyy(all_anchors_xywh)

    # process over batches
    # 逐个图像进行操作
    for b in range(bsize):
        # 确认每个图像有效的标注框数目
        num_obj = num_boxes_batch[b].item()
        # 获取YOLOv2预测结果
        # [H*W*Num_anchors, 4]
        delta_pred = delta_pred_batch[b]
        # 获取标注框数据
        # gt_box: [x1, y1, x2, y2]
        # gt_boxes: [num_obj, 4]
        gt_boxes = gt_boxes_batch[b][:num_obj, :]
        # 获取标注框对应类别下标
        gt_classes = gt_classes_batch[b][:num_obj]

        # rescale ground truth boxes
        # 缩放到特征图大小
        gt_boxes[:, 0::2] *= W
        gt_boxes[:, 1::2] *= H

        # step 1: process IoU target
        # apply delta_pred to pre-defined anchors
        #
        # 结合预测结果和锚点框得到预测框
        # all_anchors_xywh: [H*W*Num_anchors, 4]
        all_anchors_xywh = all_anchors_xywh.view(-1, 4)
        # [H*W*Num_anchors, 4]
        box_pred = box_transform_inv(all_grid_xywh, delta_pred)
        # [xc, yc, w, h] -> [x1, y1, x2, y2]
        box_pred = xywh2xxyy(box_pred)

        # for each anchor, its iou target is corresponded to the max iou with any gt boxes
        # 计算预测框和标注框之间的iou, 最大的预测框(并且IoU超过了阈值)负责该标注框的训练
        #
        # [H*W*Num_anchors, 4], [num_obj, 4] -> [H*W*Num_anchors, num_obj]
        ious = box_ious(box_pred, gt_boxes)  # shape: (H * W * num_anchors, num_obj)
        # [H*W*Num_anchors, num_obj] -> [H*W, Num_anchors, num_obj]
        ious = ious.view(-1, num_anchors, num_obj)
        # 计算每个网格中和标注框最大的iou
        max_iou, _ = torch.max(ious, dim=-1, keepdim=True)  # shape: (H * W, num_anchors, 1)
        if cfg.debug:
            print('ious', ious)

        # iou_target[b] = max_iou

        # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
        # 对于正样本(iou大于阈值), 不参与计算
        # [H*W, Num_anchors, 1] -> [H*W*Num_anchors]
        iou_thresh_filter = max_iou.view(-1) > cfg.thresh
        n_pos = torch.nonzero(iou_thresh_filter).numel()

        if n_pos > 0:
            # 如果存在, 那么不参与损失计算
            iou_mask[b][max_iou >= cfg.thresh] = 0

        # step 2: process box target and class target
        # calculate overlaps between anchors and gt boxes
        #
        # 如何确定正样本和负样本???
        # 应该是锚点框和真值框进行匹配, 哪个锚点框负责真值框预测, 预测框的目的是为了让锚点框更好的拟合标注框!!!
        # [H*W*Num_anchors, 4], [num_obj, 4] -> [H*W*Num_anchors, num_obj] -> [H*W, Num_anchors, num_obj]
        overlaps = box_ious(all_anchors_xxyy, gt_boxes).view(-1, num_anchors, num_obj)
        gt_boxes_xywh = xxyy2xywh(gt_boxes)

        # iterate over all objects
        # 每个标注框选择一个锚点框进行训练
        for t in range(gt_boxes.size(0)):
            # compute the center of each gt box to determine which cell it falls on
            # assign it to a specific anchor by choosing max IoU
            # 首先计算锚点框的中心点位于哪个网格, 然后选择其中IoU最大的锚点框参与训练

            # 第t个锚点框
            gt_box_xywh = gt_boxes_xywh[t]
            # 对应的类别下标
            gt_class = gt_classes[t]
            # 对应网格下标
            cell_idx_x, cell_idx_y = torch.floor(gt_box_xywh[:2])
            # 网格列表下标
            cell_idx = cell_idx_y * W + cell_idx_x
            cell_idx = cell_idx.long()

            # update box_target, box_mask
            # 获取该标注框在对应网格上与所有锚点框的IoU
            # [H*W, Num_anchors, num_obj] -> [Num_anchors]
            overlaps_in_cell = overlaps[cell_idx, :, t]
            # 选择IoU最大的锚点框下标
            argmax_anchor_idx = torch.argmax(overlaps_in_cell)

            # [H*W*Num_anchors, 4] -> [H*W, Num_anchors, 4] -> [4] -> [1, 4]
            # 获取对应网格中指定锚点框的坐标 [x1, y1, w, h]
            assigned_grid = all_grid_xywh.view(-1, num_anchors, 4)[cell_idx, argmax_anchor_idx, :].unsqueeze(0)
            # [4] -> [1, 4]
            gt_box = gt_box_xywh.unsqueeze(0)
            # 锚点框和标注框之间的差距就是YOLOv2需要学习的偏移
            target_t = box_transform(assigned_grid, gt_box)
            if cfg.debug:
                print('assigned_grid, ', assigned_grid)
                print('gt: ', gt_box)
                print('target_t, ', target_t)
            # 赋值, 对应掩码下标设置为1
            box_target[b, cell_idx, argmax_anchor_idx, :] = target_t.unsqueeze(0)
            box_mask[b, cell_idx, argmax_anchor_idx, :] = 1

            # update cls_target, cls_mask
            # 赋值对应类别下标, 对应掩码设置为1
            class_target[b, cell_idx, argmax_anchor_idx, :] = gt_class
            class_mask[b, cell_idx, argmax_anchor_idx, :] = 1

            # update iou target and iou mask
            iou_target[b, cell_idx, argmax_anchor_idx, :] = max_iou[cell_idx, argmax_anchor_idx, :]
            if cfg.debug:
                print(max_iou[cell_idx, argmax_anchor_idx, :])
            iou_mask[b, cell_idx, argmax_anchor_idx, :] = cfg.object_scale

    # iou_mask: 置信度掩码
    # box_mask: 预测框掩码
    # class_mask: 类别下标
    #
    # 类别下标, 关键是

    return iou_target.view(bsize, -1, 1), \
        iou_mask.view(bsize, -1, 1), \
        box_target.view(bsize, -1, 4), \
        box_mask.view(bsize, -1, 1), \
        class_target.view(bsize, -1, 1).long(), \
        class_mask.view(bsize, -1, 1)


def yolo_loss(output, target):
    """
    Build yolo loss

    计算YOLOv2损失, 包含3部分:
    1. 边界框损失: 仅计算负责标注框预测的锚点框的预测框坐标损失
    2. 置信度损失
    3. 分类损失

    Arguments:
    output -- tuple (delta_pred, conf_pred, class_score), output data of the yolo network
    target -- tuple (iou_target, iou_mask, box_target, box_mask, class_target, class_mask) target label data

    delta_pred -- Variable of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
    conf_pred -- Variable of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
    class_score -- Variable of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2 ..)

    iou_target -- Variable of shape (B, H * W * num_anchors, 1)
    iou_mask -- Variable of shape (B, H * W * num_anchors, 1)
    box_target -- Variable of shape (B, H * W * num_anchors, 4)
    box_mask -- Variable of shape (B, H * W * num_anchors, 1)
    class_target -- Variable of shape (B, H * W * num_anchors, 1)
    class_mask -- Variable of shape (B, H * W * num_anchors, 1)

    Return:
    loss -- yolo overall multi-task loss
    """

    delta_pred_batch = output[0]
    conf_pred_batch = output[1]
    class_score_batch = output[2]

    """
    创建两个部分:
    1. target: 真值标签, 负责计算损失. 
    2. mask: 不是所有的数据都参与运算. 比如边界框预测仅计算负责标注框预测的锚点框对应的预测框损失
    """
    iou_target = target[0]
    iou_mask = target[1]
    box_target = target[2]
    box_mask = target[3]
    class_target = target[4]
    class_mask = target[5]

    b, _, num_classes = class_score_batch.size()
    # [B, H * W * num_anchors, num_classes] -> [B * H * W * num_anchors, num_classes]
    class_score_batch = class_score_batch.view(-1, num_classes)
    # [B, H * W * num_anchors, 1] -> [B * H * W * num_anchors]
    class_target = class_target.view(-1)
    # [B, H * W * num_anchors, 1] -> [B * H * W * num_anchors]
    class_mask = class_mask.view(-1)

    # ignore the gradient of noobject's target
    class_keep = class_mask.nonzero().squeeze(1)
    class_score_batch_keep = class_score_batch[class_keep, :]
    class_target_keep = class_target[class_keep]

    # if cfg.debug:
    #     print(class_score_batch_keep)
    #     print(class_target_keep)

    # calculate the loss, normalized by batch size.
    box_loss = 1 / b * cfg.coord_scale * F.mse_loss(delta_pred_batch * box_mask, box_target * box_mask,
                                                    reduction='sum') / 2.0
    iou_loss = 1 / b * F.mse_loss(conf_pred_batch * iou_mask, iou_target * iou_mask, reduction='sum') / 2.0
    class_loss = 1 / b * cfg.class_scale * F.cross_entropy(class_score_batch_keep, class_target_keep, reduction='sum')

    return box_loss, iou_loss, class_loss


if __name__ == '__main__':
    grid_h = 13
    grid_w = 13
    num_anchors = 5
    N = 4
    num_classes = 20

    delta_pred_batch = torch.randn(N, grid_h * grid_w * num_anchors, 4)
    conf_pred_batch = torch.randn(N, grid_h * grid_w * num_anchors, 1)
    class_score_batch = torch.randn(N, grid_h * grid_w * num_anchors, num_classes)

    NUM_MAX_BOXES = 30
    gt_boxes_batch = torch.randn(N, NUM_MAX_BOXES, 4)
    gt_classes_batch = torch.randn(N, NUM_MAX_BOXES)
    num_obj_batch = torch.randn(N, 1)

    output = (delta_pred_batch, conf_pred_batch, class_score_batch)
    gt_data = (gt_boxes_batch, gt_classes_batch, num_obj_batch)
    # 基于输出结果和标注框数据构建target
    target_data = build_target(output, gt_data, grid_h, grid_w)
    print(target_data)
