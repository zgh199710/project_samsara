import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from samsara.datasets import Dataset
from samsara import DataLoader
from samsara.transforms import Compose, ToFloat, Normalize
from samsara.utils import get_file, cache_dir
from yolo_demo.config.dataset_config import dataset_cfg

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
)


class VOCDetection(Dataset):
    def __init__(self,
                 img_size=640,
                 data_dir=None,
                 image_sets=[('2007', 'trainval')],
                 trans_config=None,
                 transform=Compose([ToFloat(), Normalize(mean=0.5, std=0.5)]),
                 is_train=True,
                 load_cache=False):
        self.root = data_dir
        self.img_size = img_size
        self.image_sets = image_sets
        self.trans_config = trans_config
        self.is_train = is_train
        self.load_cache = load_cache
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = []

        super().__init__(train=is_train, transform=transform)

    def prepare(self):
        for (year, name) in self.image_sets:
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

        self.mosaic_prob = self.trans_config['mosaic_prob'] if self.trans_config else 0.0

        print('==============================')
        print('Use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('==============================')

        if self.load_cache:
            self._load_cache()

    def _load_cache(self):
        # 实现缓存加载逻辑
        pass

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        if np.random.random() < self.mosaic_prob:
            image, target = self.load_mosaic(index)
        else:
            image, target = self.load_image_target(index)

        if image is None or target is None:
            # Fallback to a simple image if everything fails
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            target = {"boxes": np.array([]), "labels": np.array([]), "orig_size": [self.img_size, self.img_size]}

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def load_image_target(self, index):
        img_id = self.ids[index]
        image = cv2.imread(self._imgpath % img_id)
        if image is None:
            print(f"Failed to load image: {self._imgpath % img_id}")
            return None, None

        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, (self.img_size, self.img_size))

        target = self.parse_voc_xml(ET.parse(self._annopath % img_id).getroot())

        # Resize bounding boxes
        if len(target['boxes']) > 0:
            boxes = np.array(target['boxes'], dtype=np.float32)
            boxes[:, [0, 2]] *= (self.img_size / orig_w)
            boxes[:, [1, 3]] *= (self.img_size / orig_h)
            target['boxes'] = boxes

        target['labels'] = np.array(target['labels'], dtype=np.int64)
        target['orig_size'] = [orig_h, orig_w]

        return image, target

    def parse_voc_xml(self, node):
        target = {'boxes': [], 'labels': []}
        for obj in node.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            bndbox = [int(bbox.find(pt).text) - 1 for pt in ['xmin', 'ymin', 'xmax', 'ymax']]
            label_idx = VOC_CLASSES.index(name)
            target['boxes'].append(bndbox)
            target['labels'].append(label_idx)
        return target

    def load_mosaic(self, index):
        indices = [index] + [np.random.randint(0, len(self.ids)) for _ in range(3)]
        images, targets = [], []
        for idx in indices:
            img, target = self.load_image_target(idx)
            if img is not None and target is not None:
                images.append(img)
                targets.append(target)

        if len(images) == 0:
            return self.load_image_target(index)  # Fallback to normal loading

        mosaic_img = np.zeros((self.img_size * 2, self.img_size * 2, 3), dtype=np.uint8)
        for i, img in enumerate(images):
            x = i % 2
            y = i // 2
            x1, y1 = int(self.img_size * x), int(self.img_size * y)
            x2, y2 = x1 + self.img_size, y1 + self.img_size
            mosaic_img[y1:y2, x1:x2] = img

        # Resize mosaic image to original size
        mosaic_img = cv2.resize(mosaic_img, (self.img_size, self.img_size))

        mosaic_target = {
            "boxes": [],
            "labels": []
        }

        for i, target in enumerate(targets):
            boxes = target["boxes"]
            labels = target["labels"]
            x_offset = (i % 2) * 0.5
            y_offset = (i // 2) * 0.5

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                x1 = x1 * 0.5 + x_offset
                y1 = y1 * 0.5 + y_offset
                x2 = x2 * 0.5 + x_offset
                y2 = y2 * 0.5 + y_offset
                mosaic_target["boxes"].append([x1, y1, x2, y2])
                mosaic_target["labels"].append(label)

        mosaic_target["boxes"] = np.array(mosaic_target["boxes"], dtype=np.float32)
        mosaic_target["labels"] = np.array(mosaic_target["labels"], dtype=np.int64)
        mosaic_target["orig_size"] = [self.img_size, self.img_size]

        return mosaic_img, mosaic_target

    @staticmethod
    def labels():
        return {i: name for i, name in enumerate(VOC_CLASSES)}

    def show(self, row=3, col=3):
        canvas = np.zeros((self.img_size * row, self.img_size * col, 3), dtype=np.uint8)

        for i in range(row):
            for j in range(col):
                idx = np.random.randint(0, len(self))
                img, target = self[idx]

                # 确保图像是 HWC 格式
                if isinstance(img, np.ndarray):
                    if img.shape[0] == 3:
                        img = img.transpose(1, 2, 0)  # CHW to HWC

                # 将图像值范围调整到 0-255
                img = (img * 255).astype(np.uint8)

                # 转换为BGR格式（OpenCV使用BGR）
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # 绘制边界框和标注信息
                for box, label in zip(target['boxes'], target['labels']):
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    class_name = VOC_CLASSES[label]  # 假设VOC_CLASSES是类别名称的列表

                    # 绘制边界框
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 添加标注文本
                    label_size, baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    y1 = max(y1, label_size[1])
                    cv2.rectangle(img, (x1, y1 - label_size[1] - baseline), (x1 + label_size[0], y1), (255, 255, 255),
                                  cv2.FILLED)
                    cv2.putText(img, class_name, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # 将图像放入画布
                canvas[i * self.img_size:(i + 1) * self.img_size, j * self.img_size:(j + 1) * self.img_size] = img

        # 显示画布
        cv2.imshow('VOC Dataset Samples', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def build_dataset(args, data_cfg, trans_config, transform, is_train=False):
    dataset = VOCDetection(
        img_size=args.img_size,
        data_dir=os.path.join(args.root, data_cfg['data_name']),
        image_sets=[('2007', 'trainval')] if is_train else [('2007', 'val')],
        trans_config=trans_config,
        transform=transform,
        is_train=is_train,
        load_cache=args.load_cache
    )

    dataset_info = {
        'num_classes': data_cfg['num_classes'],
        'class_names': data_cfg['class_names'],
        'class_indexs': data_cfg['class_indexs']
    }

    return dataset, dataset_info


def test_voc_dataset():
    import argparse
    parser = argparse.ArgumentParser(description='VOC Dataset Test')
    parser.add_argument('--root', default='./', help='Dataset root directory')
    parser.add_argument('--img_size', default=640, type=int, help='Image size')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--load_cache', action="store_true", default=False, help='Load cached data')
    args = parser.parse_args()

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

    # 测试数据加载
    for i, (images, targets) in enumerate(train_loader):
        print(f"Batch {i + 1}:")
        print(f"  Image shape: {images.shape}")
        print(f"  Number of targets: {len(targets)}")
        print(f"  Target keys: {targets[0].keys() if isinstance(targets[0], dict) else 'Not a dict'}")

        if i == 0:
            # 显示第一个批次的图像
            print("显示数据集样本:")
            voc_train.show()

        if i >= 2:  # 只测试前几个批次
            break

    print("数据集加载测试完成！")


if __name__ == "__main__":
    test_voc_dataset()