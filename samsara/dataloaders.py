import math
pil_available = True
try:
    from PIL import Image
except:
    pil_available = False
import numpy as np
from samsara import cuda, Variable


class DataLoader:
    """数据加载器类，用于批量加载数据。

       参数:
           dataset (iterable): 包含数据的数据集。
           batch_size (int): 每批数据的大小。
           shuffle (bool): 是否在每个 epoch 随机打乱数据。默认为 True。
           gpu (bool): 是否使用 GPU。默认为 False。
       """
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.gpu = gpu

        self.reset()

    def reset(self):
        """重置迭代器状态。
        将迭代计数重置为0，并根据 shuffle 参数决定是否重新打乱数据索引。
        """
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        """返回下一批数据。

        返回:
            tuple: 包含两个元素的元组：
                - x (numpy.ndarray 或 cupy.ndarray): 批量输入数据。
                - t (numpy.ndarray 或 cupy.ndarray): 批量目标数据。

        异常:
            StopIteration: 当遍历完所有数据后抛出。
        """
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True


class SeqDataLoader(DataLoader):
    """序列数据加载器类，用于加载序列数据。

      参数:
          dataset (iterable): 包含序列数据的数据集。
          batch_size (int): 每批数据的大小。
          gpu (bool): 是否使用 GPU。默认为 False。
      """
    def __init__(self, dataset, batch_size, gpu=False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False,
                         gpu=gpu)

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        jump = self.data_size // self.batch_size
        batch_index = [(i * jump + self.iteration) % self.data_size for i in
                       range(self.batch_size)]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t