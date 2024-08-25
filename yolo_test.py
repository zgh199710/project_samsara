from yolo_demo.build import build_yolov1
from yolo_demo.config import yolov1_cfg, dataset_cfg
import argparse
import numpy as np
from samsara import Variable, optimizers, datasets, DataLoader
from samsara.transforms import Compose, ToFloat, Normalize
from voc import build_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv1')
    # Basic
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use cuda.')
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='input image size')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')

    parser.add_argument('--save_folder', default='weights/', type=str,
                        help='path to save weight')
    parser.add_argument('--eval_first', action='store_true', default=False,
                        help='evaluate model before training.')
    # Batchsize
    parser.add_argument('-bs', '--batch_size', default=2, type=int,
                        help='batch size on all the GPUs.')

    # Epoch
    parser.add_argument('--max_epoch', default=20, type=int,
                        help='max epoch.')
    parser.add_argument('--wp_epoch', default=1, type=int,
                        help='warmup epoch.')
    parser.add_argument('--eval_epoch', default=10, type=int,
                        help='after eval epoch, the model is evaluated on val dataset.')
    parser.add_argument('--no_aug_epoch', default=20, type=int,
                        help='cancel strong augmentation.')

    # Model
    parser.add_argument('-m', '--model', default='yolov1', type=str,
                        help='build yolo_demo')
    parser.add_argument('-ct', '--conf_thresh', default=0.005, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='topk candidates for evaluation')
    parser.add_argument('-p', '--pretrained', default=None, type=str,
                        help='load pretrained weight')
    parser.add_argument('-r', '--resume', default="weights/voc/yolov1/yolov1_best.pth", type=str,
                        help='keep training')

    # Dataset
    parser.add_argument('--root', default='./', help='Dataset root directory')

    parser.add_argument('--load_cache', action="store_true", default=False, help='Load cached data')


    return parser.parse_args()

def test():
    args = parse_args()
    batch_size = 2
    channels = 3
    height = args.img_size
    width = args.img_size

    num_classes = 20  # 根据数据集修改
    model, criterion = build_yolov1(args, yolov1_cfg, num_classes, trainable=True)
    dir(model)

    lr = 1e-2
    optimizer = optimizers.SGD(lr).setup(model)
    max_epoch = 100
    batch_size = 30

    # 设置转换和增强配置
    transform = Compose([ToFloat(), Normalize(mean=0.5, std=0.5)])
    trans_config = {
        'mosaic_prob': 0.5,
        'mixup_prob': 0.5,
    }

    # 构建数据集
    voc_train, train_info = build_dataset(args, dataset_cfg['voc'], trans_config, transform, is_train=True)
    voc_test, test_info = build_dataset(args, dataset_cfg['voc'], trans_config, transform, is_train=False)

    print(f"训练集大小: {len(voc_train)}")
    print(f"测试集大小: {len(voc_test)}")
    print(f"类别数量: {train_info['num_classes']}")
    print(f"类别名称: {train_info['class_names']}")

    # 创建数据加载器
    train_loader = DataLoader(voc_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(voc_test, batch_size=args.batch_size, shuffle=False)

    for epoch in range(max_epoch):
        for i, (images, targets) in enumerate(train_loader):
            images = images.transpose(0, 3, 1, 2)
            y = model(images)

            # # 打印一些调试信息
            # print("Model output shapes:")
            # for key, value in y.items():
            #     if isinstance(value, Variable):
            #         print(f"{key}: {value.shape}")
            #     else:
            #         print(f"{key}: {value}")

            # 确保 targets 的格式正确
            formatted_targets = []
            for target in targets:
                formatted_target = {
                    'labels': target['labels'],
                    'boxes': target['boxes']
                }
                formatted_targets.append(formatted_target)

            # # 打印一些目标的调试信息
            # print("Target shapes:")
            # for target in formatted_targets:
            #     print(f"labels: {target['labels'].shape}, boxes: {target['boxes'].shape}")

            loss_dict, total_loss = criterion(y, formatted_targets)

            print("Loss values:")
            for key, value in loss_dict.items():
                print(f"{key}: {value.data}")

            model.cleargrads()
            total_loss.backward()
            optimizer.update()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Total loss: {total_loss.data}")

if __name__ == '__main__':
    test()

