import numpy as np
from samsara import Variable
# from .yolo_loss import build_criterion
from .loss import build_criterion
from yolo_demo.yolo import YOLOv1


def build_yolov1(args, cfg, num_classes=80, trainable=False, deploy=False):
    print('==============================')
    print('Build {} ...'.format(args.model.upper()))

    print('==============================')
    print('Model Configuration: \n', cfg)

    model = YOLOv1(
        cfg=cfg,
        img_size=args.img_size,
        num_classes=num_classes,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        trainable=trainable,
        deploy=deploy
    )

    # 打印模型结构
    print('==============================')
    print('Model Structure:')
    print(model)

    # Init bias
    init_prob = 0.01
    bias_value = -np.log((1. - init_prob) / init_prob)
    # obj pred
    b = model.obj_pred.b.data.reshape(1, -1)
    b.fill(bias_value)
    model.obj_pred.b.data = b.reshape(-1)
    # cls pred
    b = model.cls_pred.b.data.reshape(1, -1)
    b.fill(bias_value)
    model.cls_pred.b.data = b.reshape(-1)
    # reg pred
    b = model.reg_pred.b.data
    b.fill(1.0)
    model.reg_pred.b.data = b
    w = model.reg_pred.W.data
    w.fill(0.)
    model.reg_pred.W.data = w

    criterion = None
    if trainable:
        criterion = build_criterion(cfg, num_classes)

    return model, criterion


if __name__ == '__main__':
    build_yolov1()