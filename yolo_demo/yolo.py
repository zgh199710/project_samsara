import numpy as np
from samsara import Variable, as_variable, Layer
import samsara.functions as F
import samsara.layers as L
from samsara.models import Model, Sequential

from .yolo_backbone import build_backbone
from .yolo_head import build_head
from .yolo_neck import build_neck


def nms(bboxes, scores, nms_thresh):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 2]

    areas = (x2 - x1) * (y2 - y1)
    '''
    目的是对 scores 数组进行降序排序，并返回排序后的索引。具体地，scores.argsort()[::-1] 的含义如下：
    scores.argsort(): 返回一个索引数组，这个数组中的索引按照 scores 中对应元素的大小升序排列。
    [::-1]: 对这个索引数组进行反转，以得到降序排列的索引数组。
    假设 scores 是一个包含分数的 NumPy 数组，这句代码返回 scores 从大到小排列时的索引。
    '''
    # order = scores.argsort()[::-1]
    order = np.argsort(-scores)  # 更简洁的降序排序
    keep = [] # 存储保留下来的边界框索引
    while order.size > 0:
        i = order[0] # 当前分数最高的边界框索引
        keep.append(i) # 将该索引加入保留列表

        # 计算交集区域的左上角和右下角的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 防止除零错误
        # 在计算过程中，如果某个值为零，那么进行除法操作时就会出现除零错误。为了避免这种情况，我们通常会加上一个非常小的数，确保分母永远不为零。
        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        # 选择IoU小于阈值的边界框索引
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


## class-agnostic NMS
'''
类别无关的NMS (Class-agnostic NMS)
类别无关的NMS不考虑边界框的类别标签，只根据边界框的分数进行NMS操作。
'''

def multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh):
    # nms
    keep = nms(bboxes, scores, nms_thresh)

    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes


## class-aware NMS
'''
类别相关的NMS (Class-aware NMS)
类别相关的NMS会根据每个类别分别进行NMS操作，然后合并结果。
'''
def multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes):
    # nms
    keep = np.zeros(len(bboxes), dtype=np.int32)
    for i in range(num_classes):
        inds = np.where(labels == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores, nms_thresh)
        keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes

## multi-class NMS
def multiclass_nms(scores, labels, bboxes, nms_thresh, num_classes, class_agnostic=False):
    if class_agnostic:
        return multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh)
    else:
        return multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes)


class YOLOv1(Model):
    def __init__(self,
                 cfg,
                 img_size=None,
                 num_classes=80,
                 conf_thresh=0.01,
                 nms_thresh=0.5,
                 trainable=False,
                 deploy=False):
        super(YOLOv1, self).__init__()

        self.cfg = cfg
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = 32
        self.deploy = deploy

        self.backbone, feat_dim = build_backbone(
            cfg['backbone'], trainable and cfg['pretrained']
        )

        self.neck = build_neck(
            cfg, feat_dim, out_dim=512
        )
        head_dim = self.neck.out_dim

        self.head = build_head(cfg, head_dim, head_dim, num_classes)

        self.obj_pred = L.Conv2d(out_channels=1, kernel_size=1, in_channels=head_dim)
        self.cls_pred = L.Conv2d(out_channels=num_classes, kernel_size=1, in_channels=head_dim)
        self.reg_pred = L.Conv2d(out_channels=4, kernel_size=1, in_channels=head_dim)

    def inference(self, x):
        feat = self.backbone(x)
        feat = self.neck(feat)
        cls_feat, reg_feat = self.head(feat)

        obj_pred = self.obj_pred(cls_feat)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        img_size = obj_pred.shape[-2:]

        obj_pred = F.transpose(obj_pred, (0, 2, 3, 1))
        obj_pred = F.reshape(obj_pred, (obj_pred.shape[0], -1, 1))
        cls_pred = F.transpose(cls_pred, (0, 2, 3, 1))
        cls_pred = F.reshape(cls_pred, (cls_pred.shape[0], -1, self.num_classes))
        reg_pred = F.transpose(reg_pred, (0, 2, 3, 1))
        reg_pred = F.reshape(reg_pred, (reg_pred.shape[0], -1, 4))

        obj_pred = obj_pred[0]
        cls_pred = cls_pred[0]
        reg_pred = reg_pred[0]

        scores = F.sqrt(F.sigmoid(obj_pred) * F.sigmoid(cls_pred))

        bboxes = self.decode_boxes(reg_pred, img_size)

        if self.deploy:
            outputs = F.concat([bboxes, scores], axis=-1)
            return outputs
        else:
            scores = scores.data
            bboxes = bboxes.data

            bboxes, scores, labels = self.postprocess(bboxes, scores)

        return bboxes, scores, labels

    def create_grid(self, img_size):
        ws, hs = img_size
        grid_y, grid_x = np.meshgrid(np.arange(hs), np.arange(ws), indexing='ij')
        grid_xy = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
        grid_xy = grid_xy.reshape(-1, 2)
        return as_variable(grid_xy)

    def decode_boxes(self, pred_reg, img_size):
        # 生成网格坐标矩阵
        grid_cell = self.create_grid(img_size)

        # 计算预测边界框的中心点坐标和宽高
        pred_ctr = (F.sigmoid(pred_reg[..., :2]) + grid_cell) * self.stride
        pred_wh = F.exp(pred_reg[..., 2:]) * self.stride

        # 将所有bbox的中心点坐标和宽高换算成x1y1x2y2形式
        pred_x1y1 = pred_ctr - pred_wh * 0.5
        pred_x2y2 = pred_ctr + pred_wh * 0.5
        pred_box = F.concat((pred_x1y1, pred_x2y2), axis=-1)

        return pred_box

    def forward(self, x):
        if not self.trainable:
            return self.inference(x)
        else:
            feat = self.backbone(x)
            feat = self.neck(feat)
            cls_feat, reg_feat = self.head(feat)

            obj_pred = self.obj_pred(cls_feat)
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)
            img_size = obj_pred.shape[-2:]

            obj_pred = F.transpose(obj_pred, (0, 2, 3, 1))
            obj_pred = F.reshape(obj_pred, (obj_pred.shape[0], -1, 1))
            cls_pred = F.transpose(cls_pred, (0, 2, 3, 1))
            cls_pred = F.reshape(cls_pred, (cls_pred.shape[0], -1, self.num_classes))
            reg_pred = F.transpose(reg_pred, (0, 2, 3, 1))
            reg_pred = F.reshape(reg_pred, (reg_pred.shape[0], -1, 4))

            box_pred = self.decode_boxes(reg_pred, img_size)

            outputs = {
                "pred_obj": obj_pred,
                "pred_cls": cls_pred,
                "pred_box": box_pred,
                "stride": self.stride,
                "img_size": img_size
            }

            return outputs

    def postprocess(self, bboxes, scores):
        """
            后处理环节，包括<阈值筛选>和<非极大值抑制(NMS)>两个环节
            输入:
                bboxes: (numpy.array) -> [HxW, 4]
                scores: (numpy.array) -> [HxW, num_classes]
            输出:
                bboxes: (numpy.array) -> [N, 4]
                score:  (numpy.array) -> [N,]
                labels: (numpy.array) -> [N,]
        """
        labels = np.argmax(scores, axis=1)
        '''
        这个错误通常是因为索引方式不正确。需要确保 np.arange 和 labels 之间的操作是正确的。
        假设你的原始代码是：
        scores = scores[np.arange(scores.shape[0], labels)]
        这行代码的意图可能是根据 labels 中的索引获取 scores。应该检查 labels 是否是一个有效的索引数组。通常情况下，应该是：
        scores = scores[np.arange(scores.shape[0]), labels]
        这种写法确保了你在 scores 中获取每个元素的正确索引。
        '''
        scores = scores[np.arange(scores.shape[0]), labels]

        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, False)

        return bboxes, scores, labels
