import math
from samsara import cuda, Parameter


# =============================================================================
# Optimizer (base class)
# =============================================================================
class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)


# =============================================================================
# Hook functions
# =============================================================================
class WeightDecay:
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, params):
        for param in params:
            param.grad.data += self.rate * param.data


class ClipGrad:
    def __init__(self, max_norm):
        self.max_norm = max_norm

    def __call__(self, params):
        total_norm = 0
        for param in params:
            total_norm += (param.grad.data ** 2).sum()
        total_norm = math.sqrt(float(total_norm))

        rate = self.max_norm / (total_norm + 1e-6)
        if rate < 1:
            for param in params:
                param.grad.data *= rate


class FreezeParam:
    def __init__(self, *layers):
        self.freeze_params = []
        for l in layers:
            if isinstance(l, Parameter):
                self.freeze_params.append(l)
            else:
                for p in l.params():
                    self.freeze_params.append(p)

    def __call__(self, params):
        for p in self.freeze_params:
            p.grad = None



# =============================================================================
# SGD / MomentumSGD / AdaGrad / AdaDelta / Adam
# =============================================================================
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            xp = cuda.get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v


class AdaGrad(Optimizer):
    def __init__(self, lr=0.001, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = xp.zeros_like(param.data)

        lr = self.lr
        eps = self.eps
        grad = param.grad.data
        h = self.hs[h_key]

        h += grad * grad
        param.data -= lr * grad / (xp.sqrt(h) + eps)


class AdaDelta(Optimizer):
    def __init__(self, rho=0.95, eps=1e-6):
        super().__init__()
        self.rho = rho
        self.eps = eps
        self.msg = {}
        self.msdx = {}

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.msg:
            self.msg[key] = xp.zeros_like(param.data)
            self.msdx[key] = xp.zeros_like(param.data)

        msg, msdx = self.msg[key], self.msdx[key]
        rho = self.rho
        eps = self.eps
        grad = param.grad.data

        msg *= rho
        msg += (1 - rho) * grad * grad
        dx = xp.sqrt((msdx + eps) / (msg + eps)) * grad
        msdx *= rho
        msdx += (1 - rho) * dx * dx
        param.data -= dx


class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        param.data -= self.lr * m / (xp.sqrt(v) + eps)


class AdamW(Optimizer):
    """AdamW 优化器。

    AdamW 是 Adam 的一个变体，实现了修正的权重衰减。

    Args:
        alpha (float): 学习率
        beta1 (float): 一阶矩估计的指数衰减率
        beta2 (float): 二阶矩估计的指数衰减率
        eps (float): 数值稳定性常数
        weight_decay (float): 权重衰减（L2惩罚）系数
    """

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)

        # AdamW 的权重衰减实现
        param.data -= self.weight_decay * self.alpha * param.data
        param.data -= self.lr * m / (xp.sqrt(v) + eps)


class RAdam(Optimizer):
    """Rectified Adam 优化器。

    RAdam 通过在训练早期阶段自适应地调整学习率来改进 Adam。

    Args:
        alpha (float): 学习率
        beta1 (float): 一阶矩估计的指数衰减率
        beta2 (float): 二阶矩估计的指数衰减率
        eps (float): 数值稳定性常数
    """

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)

        # RAdam 的学习率调整
        rho_inf = 2 / (1 - beta2) - 1
        rho_t = rho_inf - 2 * self.t * beta2 ** self.t / (1 - beta2 ** self.t)

        if rho_t > 4:
            r_t = math.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
            var = xp.sqrt(v / (1 - beta2 ** self.t)) + eps
            param.data -= self.alpha * r_t * m / (var * (1 - beta1 ** self.t))
        else:
            param.data -= self.alpha * m / (1 - beta1 ** self.t)


class Lookahead(Optimizer):
    """Lookahead 优化器。

    Lookahead 是一个优化器包装器，可以应用于任何优化算法。

    Args:
        optimizer (Optimizer): 基础优化器
        k (int): 每 k 步更新一次慢权重
        alpha (float): 慢权重更新的步长
    """

    def __init__(self, optimizer, k=5, alpha=0.5):
        super().__init__()
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.slow_weights = {}
        self.counter = 0

    def update(self):
        self.counter += 1
        self.optimizer.update()

        if self.counter >= self.k:
            self.counter = 0
            for param in self.target.params():
                if param.grad is None:
                    continue
                key = id(param)
                if key not in self.slow_weights:
                    self.slow_weights[key] = param.data.copy()
                slow = self.slow_weights[key]
                slow += self.alpha * (param.data - slow)
                param.data[:] = slow

    def update_one(self, param):
        # 这个方法在 Lookahead 中不使用，所有更新都在 update 方法中完成
        pass


class Ranger(Optimizer):
    """Ranger 优化器。

    Ranger 结合了 RAdam 和 Lookahead 的优点。

    Args:
        alpha (float): 学习率
        k (int): Lookahead 的步数
        N_sma_threshhold (int): SMA 阈值
        betas (tuple): Adam 的 beta 参数
        eps (float): 数值稳定性常数
        weight_decay (float): 权重衰减系数
    """

    def __init__(self, alpha=0.001, k=6, N_sma_threshhold=5, betas=(0.95, 0.999), eps=1e-5, weight_decay=0):
        super().__init__()
        self.alpha = alpha
        self.k = k
        self.N_sma_threshhold = N_sma_threshhold
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # RAdam 组件
        self.radam = RAdam(alpha=alpha, beta1=betas[0], beta2=betas[1], eps=eps)

        # Lookahead 组件
        self.slow_weights = {}
        self.counter = 0

    def update(self):
        self.counter += 1
        self.radam.update()

        if self.counter >= self.k:
            self.counter = 0
            for param in self.target.params():
                if param.grad is None:
                    continue
                key = id(param)
                if key not in self.slow_weights:
                    self.slow_weights[key] = param.data.copy()
                slow = self.slow_weights[key]
                slow += 0.5 * (param.data - slow)
                param.data[:] = slow

    def update_one(self, param):
        # 这个方法在 Ranger 中不使用，所有更新都在 update 方法中完成
        pass


class RMSprop(Optimizer):
    """RMSprop 优化器。

    RMSprop 通过使用移动平均来调整学习率，有助于处理非平稳目标。

    Args:
        lr (float): 学习率
        alpha (float): 平滑常数
        eps (float): 添加到分母以提高数值稳定性的项
    """

    def __init__(self, lr=0.01, alpha=0.99, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.ms = {}

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)

        lr = self.lr
        eps = self.eps
        grad = param.grad.data
        ms = self.ms[key]

        ms *= self.alpha
        ms += (1 - self.alpha) * grad * grad
        param.data -= lr * grad / (xp.sqrt(ms) + eps)


class Nadam(Optimizer):
    """Nadam 优化器。

    Nadam 结合了 Adam 和 Nesterov 动量的优点。

    Args:
        lr (float): 学习率
        beta1 (float): 一阶矩估计的指数衰减率
        beta2 (float): 二阶矩估计的指数衰减率
        eps (float): 数值稳定性常数
    """

    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.m:
            self.m[key] = xp.zeros_like(param.data)
            self.v[key] = xp.zeros_like(param.data)

        m, v = self.m[key], self.v[key]
        beta1, beta2 = self.beta1, self.beta2
        t = self.t
        grad = param.grad.data

        m_t = beta1 * m + (1 - beta1) * grad
        v_t = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m_t / (1 - beta1 ** t)
        v_hat = v_t / (1 - beta2 ** t)

        m_bar = beta1 * m_hat + ((1 - beta1) / (1 - beta1 ** t)) * grad

        param.data -= self.lr * m_bar / (xp.sqrt(v_hat) + self.eps)

        self.m[key] = m_t
        self.v[key] = v_t


class AMSGrad(Optimizer):
    """AMSGrad 优化器。

    AMSGrad 是 Adam 的一个变体，旨在解决 Adam 在某些情况下可能无法收敛的问题。

    Args:
        lr (float): 学习率
        beta1 (float): 一阶矩估计的指数衰减率
        beta2 (float): 二阶矩估计的指数衰减率
        eps (float): 数值稳定性常数
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.v_hat = {}
        self.t = 0

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.m:
            self.m[key] = xp.zeros_like(param.data)
            self.v[key] = xp.zeros_like(param.data)
            self.v_hat[key] = xp.zeros_like(param.data)

        m, v, v_hat = self.m[key], self.v[key], self.v_hat[key]
        beta1, beta2 = self.beta1, self.beta2
        t = self.t
        grad = param.grad.data

        m_t = beta1 * m + (1 - beta1) * grad
        v_t = beta2 * v + (1 - beta2) * (grad ** 2)
        v_hat_t = xp.maximum(v_hat, v_t)

        m_hat = m_t / (1 - beta1 ** t)
        v_hat_hat = v_hat_t / (1 - beta2 ** t)

        param.data -= self.lr * m_hat / (xp.sqrt(v_hat_hat) + self.eps)

        self.m[key] = m_t
        self.v[key] = v_t
        self.v_hat[key] = v_hat_t


class AdaBelief(Optimizer):
    """AdaBelief 优化器。

    AdaBelief 通过考虑梯度的可信度来调整学习率，可以在各种任务中表现良好。

    Args:
        lr (float): 学习率
        beta1 (float): 一阶矩估计的指数衰减率
        beta2 (float): 二阶矩估计的指数衰减率
        eps (float): 数值稳定性常数
        weight_decay (float): 权重衰减（L2惩罚）系数
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.s = {}
        self.t = 0

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.m:
            self.m[key] = xp.zeros_like(param.data)
            self.s[key] = xp.zeros_like(param.data)

        m, s = self.m[key], self.s[key]
        beta1, beta2 = self.beta1, self.beta2
        t = self.t
        grad = param.grad.data

        if self.weight_decay != 0:
            grad = grad + self.weight_decay * param.data

        m_t = beta1 * m + (1 - beta1) * grad
        s_t = beta2 * s + (1 - beta2) * ((grad - m_t) ** 2) + self.eps

        m_hat = m_t / (1 - beta1 ** t)
        s_hat = s_t / (1 - beta2 ** t)

        param.data -= self.lr * m_hat / (xp.sqrt(s_hat) + self.eps)

        self.m[key] = m_t
        self.s[key] = s_t