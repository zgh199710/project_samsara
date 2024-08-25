import numpy as np
from samsara.core import Variable
from samsara import functions as F


class CustomVariable(Variable):
    def __getitem__(self, key):
        return CustomGetItem(key)(self)


class CustomGetItem(F.GetItem):
    def forward(self, x):
        if isinstance(x, np.ndarray):
            return x[self.slices]
        return super().forward(x)


def custom_get_item(x, slices):
    return CustomGetItem(slices)(x)


class CustomYolomatcher:
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
            target_cls = targets_per_image['labels'].data
            target_box = targets_per_image['boxes'].data

            for i in range(len(target_cls)):
                gt_label = target_cls[i]
                gt_box = target_box[i]
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

        gt_objectness = CustomVariable(gt_objectness)
        gt_classes = CustomVariable(gt_classes)
        gt_bboxes = CustomVariable(gt_bboxes)

        return gt_objectness, gt_classes, gt_bboxes