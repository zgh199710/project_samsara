import numpy as np
import samsara
import samsara.functions as F
from samsara.core import Variable
from samsara.models import Model


def get_ious(bboxes1, bboxes2, box_mode="xyxy", iou_type="iou"):
    if box_mode == "ltrb":
        bboxes1 = F.concat((-bboxes1[..., :2], bboxes1[..., 2:]), axis=-1)
        bboxes2 = F.concat((-bboxes2[..., :2], bboxes2[..., 2:]), axis=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    eps = np.finfo(np.float32).eps

    bboxes1_area = F.clip(bboxes1[..., 2] - bboxes1[..., 0], 0, float('inf')) * \
                   F.clip(bboxes1[..., 3] - bboxes1[..., 1], 0, float('inf'))
    bboxes2_area = F.clip(bboxes2[..., 2] - bboxes2[..., 0], 0, float('inf')) * \
                   F.clip(bboxes2[..., 3] - bboxes2[..., 1], 0, float('inf'))

    w_intersect = F.clip(F.minimum(bboxes1[..., 2], bboxes2[..., 2]) -
                         F.maximum(bboxes1[..., 0], bboxes2[..., 0]), 0, float('inf'))
    h_intersect = F.clip(F.minimum(bboxes1[..., 3], bboxes2[..., 3]) -
                         F.maximum(bboxes1[..., 1], bboxes2[..., 1]), 0, float('inf'))

    area_intersect = w_intersect * h_intersect
    area_union = bboxes2_area + bboxes1_area - area_intersect
    ious = area_intersect / F.clip(area_union, eps, None)

    if iou_type == "iou":
        return ious
    elif iou_type == "giou":
        g_w_intersect = F.maximum(bboxes1[..., 2], bboxes2[..., 2]) - \
                        F.minimum(bboxes1[..., 0], bboxes2[..., 0])
        g_h_intersect = F.maximum(bboxes1[..., 3], bboxes2[..., 3]) - \
                        F.minimum(bboxes1[..., 1], bboxes2[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / F.clip(ac_uion, eps, None)
        return gious
    else:
        raise NotImplementedError


class Yolomatcher:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, img_size, stride, targets):
        bs = len(targets)
        img_h, img_w = img_size
        grid_h, grid_w = img_h // stride, img_w // stride
        gt_objectness = np.zeros([bs, grid_h, grid_w, 1])
        gt_classes = np.zeros([bs, grid_h, grid_w, self.num_classes])
        gt_bboxes = np.zeros([bs, grid_h, grid_w, 4])

        for batch_index in range(bs):
            targets_per_image = targets[batch_index]
            target_cls = targets_per_image['labels']
            target_box = targets_per_image['boxes']

            if isinstance(target_cls, Variable):
                target_cls = target_cls.data
            if isinstance(target_box, Variable):
                target_box = target_box.data

            for gt_box, gt_label in zip(target_box, target_cls):
                x1, y1, x2, y2 = gt_box

                xc, yc = (x2 + x1) * 0.5, (y2 + y1) * 0.5
                bw, bh = x2 - x1, y2 - y1
                if bw < 1. or bh < 1.:
                    continue

                grid_x = int(xc / stride)
                grid_y = int(yc / stride)

                if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
                    gt_objectness[batch_index, grid_y, grid_x, 0] = 1.0
                    cls_one_hot = np.zeros(self.num_classes)
                    cls_one_hot[int(gt_label)] = 1.0
                    gt_classes[batch_index, grid_y, grid_x] = cls_one_hot
                    gt_bboxes[batch_index, grid_y, grid_x] = np.array([x1, y1, x2, y2])

        gt_objectness = gt_objectness.reshape(bs, -1, 1)
        gt_classes = gt_classes.reshape(bs, -1, self.num_classes)
        gt_bboxes = gt_bboxes.reshape(bs, -1, 4)

        gt_objectness = Variable(gt_objectness)
        gt_classes = Variable(gt_classes)
        gt_bboxes = Variable(gt_bboxes)

        # print("Matcher output shapes:")
        # print(f"gt_objectness: {gt_objectness.shape}")
        # print(f"gt_classes: {gt_classes.shape}")
        # print(f"gt_bboxes: {gt_bboxes.shape}")

        return gt_objectness, gt_classes, gt_bboxes


class Criterion(Model):
    def __init__(self, cfg, num_classes=80):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.loss_obj_weight = cfg['loss_obj_weight']
        self.loss_cls_weight = cfg['loss_cls_weight']
        self.loss_box_weight = cfg['loss_box_weight']

        self.matcher = Yolomatcher(num_classes=num_classes)

    def __call__(self, outputs, targets):
        return self.forward(outputs, targets)

    def loss_objectness(self, pred_obj, gt_obj):
        pred_obj = F.reshape(pred_obj, (-1,))
        gt_obj = F.reshape(gt_obj, (-1,))

        # 确保形状匹配
        if pred_obj.shape[0] != gt_obj.shape[0]:
            min_size = min(pred_obj.shape[0], gt_obj.shape[0])
            pred_obj = pred_obj[:min_size]
            gt_obj = gt_obj[:min_size]

        # 处理空张量的情况
        if pred_obj.shape[0] == 0 or gt_obj.shape[0] == 0:
            return Variable(np.array(0.0))  # 返回零损失

        return F.binary_cross_entropy(pred_obj, gt_obj)

    def loss_classes(self, pred_cls, gt_label):
        pred_cls = F.reshape(pred_cls, (-1, self.num_classes))
        gt_label = F.reshape(gt_label, (-1, self.num_classes))

        # 确保形状匹配
        if pred_cls.shape[0] != gt_label.shape[0]:
            min_size = min(pred_cls.shape[0], gt_label.shape[0])
            pred_cls = pred_cls[:min_size]
            gt_label = gt_label[:min_size]

        # 处理空张量的情况
        if pred_cls.shape[0] == 0 or gt_label.shape[0] == 0:
            return Variable(np.array(0.0))  # 返回零损失

        return F.binary_cross_entropy(pred_cls, gt_label)

    def loss_bboxes(self, pred_box, gt_box):
        pred_box = F.reshape(pred_box, (-1, 4))
        gt_box = F.reshape(gt_box, (-1, 4))

        # 确保形状匹配
        if pred_box.shape[0] != gt_box.shape[0]:
            min_size = min(pred_box.shape[0], gt_box.shape[0])
            pred_box = pred_box[:min_size]
            gt_box = gt_box[:min_size]

        # 处理空张量的情况
        if pred_box.shape[0] == 0 or gt_box.shape[0] == 0:
            return Variable(np.array(0.0))  # 返回零损失

        ious = get_ious(pred_box, gt_box, box_mode="xyxy", iou_type="giou")
        return F.mean(1.0 - ious)  # 使用平均值而不是总和

    def forward(self, outputs, targets):
        stride = outputs['stride']
        img_size = outputs['img_size']

        pred_obj = outputs['pred_obj']
        pred_cls = outputs['pred_cls']
        pred_box = outputs['pred_box']

        gt_objectness, gt_classes, gt_bboxes = self.matcher(img_size=img_size, stride=stride, targets=targets)

        # print("Shapes:")
        # print(f"pred_obj: {pred_obj.shape}, gt_objectness: {gt_objectness.shape}")
        # print(f"pred_cls: {pred_cls.shape}, gt_classes: {gt_classes.shape}")
        # print(f"pred_box: {pred_box.shape}, gt_bboxes: {gt_bboxes.shape}")

        pos_masks = (gt_objectness.data > 0)
        num_fgs = F.clip(F.sum(pos_masks), 1.0, None)

        loss_obj = self.loss_objectness(pred_obj, gt_objectness)
        loss_obj = F.sum(loss_obj) / num_fgs

        loss_cls = self.loss_classes(pred_cls, gt_classes)
        loss_cls = F.sum(loss_cls) / num_fgs

        loss_box = self.loss_bboxes(pred_box, gt_bboxes)
        loss_box = F.sum(loss_box) / num_fgs

        print("Individual losses:")
        print(f"loss_obj: {loss_obj.data}, loss_cls: {loss_cls.data}, loss_box: {loss_box.data}")

        losses = self.loss_obj_weight * loss_obj + \
                 self.loss_cls_weight * loss_cls + \
                 self.loss_box_weight * loss_box

        loss_dict = {
            'loss_obj': loss_obj,
            'loss_cls': loss_cls,
            'loss_box': loss_box,
            'losses': losses
        }

        return loss_dict, losses


def build_criterion(cfg, num_classes):
    return Criterion(cfg=cfg, num_classes=num_classes)


def test_loss():
    # 配置参数
    cfg = {
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 1.0
    }
    num_classes = 80
    batch_size = 2
    img_size = (416, 416)
    stride = 32
    grid_size = (img_size[0] // stride, img_size[1] // stride)

    # 创建模拟的模型输出
    pred_obj = Variable(np.random.rand(batch_size, *grid_size, 1))
    pred_cls = Variable(np.random.rand(batch_size, *grid_size, num_classes))
    pred_box = Variable(np.random.rand(batch_size, *grid_size, 4))

    outputs = {
        'stride': stride,
        'img_size': img_size,
        'pred_obj': pred_obj,
        'pred_cls': pred_cls,
        'pred_box': pred_box
    }

    # 创建模拟的目标数据
    targets = []
    for _ in range(batch_size):
        target = {
            'labels': Variable(np.array([1, 2])),  # 两个目标，类别为1和2
            'boxes': Variable(np.array([[50, 50, 100, 100], [200, 200, 300, 300]]))  # 两个边界框
        }
        targets.append(target)

    # 创建 Criterion 实例
    criterion = build_criterion(cfg, num_classes)

    # 打印形状信息
    print("pred_obj shape:", pred_obj.shape)
    print("pred_cls shape:", pred_cls.shape)
    print("pred_box shape:", pred_box.shape)

    # 计算损失
    loss_dict = criterion(outputs, targets)

    # 打印损失
    print("Objectness Loss:", loss_dict['loss_obj'].data)
    print("Classification Loss:", loss_dict['loss_cls'].data)
    print("Bounding Box Loss:", loss_dict['loss_box'].data)
    print("Total Loss:", loss_dict['losses'].data)

    # 检查损失值是否合理
    assert loss_dict['loss_obj'].data > 0, "Objectness loss should be positive"
    assert loss_dict['loss_cls'].data > 0, "Classification loss should be positive"
    assert loss_dict['loss_box'].data > 0, "Bounding box loss should be positive"
    assert loss_dict['losses'].data > 0, "Total loss should be positive"

    print("All tests passed!")

if __name__ == "__main__":
    test_loss()