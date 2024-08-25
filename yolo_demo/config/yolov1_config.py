# YOLOv1 Config

yolov1_cfg = {
    # input
    'trans_type': 'ssd',
    'multi_scale': [0.5, 1.5],
    # backbone
    'backbone': 'resnet18',
    'pretrained': False,
    'stride': 32,  # P5
    'max_stride': 32,
    # neck
    'neck': 'sppf',
    'expand_ratio': 0.5,
    'pooling_size': 5,
    'neck_act': 'leakyRelu',
    'neck_norm': 'BN',
    'neck_depthwise': False,
    # head
    'head': 'decoupled_head',
    'head_act': 'leakyRelu',
    'head_norm': 'BN',
    'num_cls_head': 2,
    'num_reg_head': 2,
    'head_depthwise': False,
    # loss weight
    'loss_obj_weight': 1.0,
    'loss_cls_weight': 1.0,
    'loss_box_weight': 5.0,
    # training configuration
    'trainer_type': 'yolov8',
}