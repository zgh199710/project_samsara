import weakref
import numpy as np
import contextlib
import samsara
import cv2
from graphviz import Digraph
from io import BytesIO
from PIL import Image


# =============================================================================
# Config
# =============================================================================
class Config:
    """
     全局配置类，用于控制反向传播和训练模式。

     Attributes:
         enable_backprop (bool): 是否启用反向传播。
         train (bool): 是否处于训练模式。
         visualize_forward (bool): 是否启用前向可视化（
         cv_window_created (bool): 给opencv调窗口用的
     """
    enable_backprop = True
    train = True
    visualize_forward = False
    cv_window_created = False


def graphviz_to_opencv(dot):
    png_data = dot.pipe(format='png')
    pil_image = Image.open(BytesIO(png_data))
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return opencv_image


@contextlib.contextmanager
def using_config(name, value):
    """
      临时更改配置的上下文管理器。

      Args:
          name (str): 要更改的配置名称。
          value: 配置的新值。
      """
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def using_visualize_forward():
    return using_config('visualize_forward', True)


def no_grad():
    """
      创建一个禁用反向传播的上下文。

      Returns:
          context manager: 禁用反向传播的上下文管理器。
      """
    return using_config('enable_backprop', False)


def test_mode():
    return using_config('train', False)


# =============================================================================
# ComputationGraph
# =============================================================================
class ComputationGraph:
    """
     计算图类，用于构建、可视化和优化计算图。

     Attributes:
         nodes (set): 图中的节点集合。
         edges (list): 图中的边列表。
         optimizations (dict): 优化选项字典。
     """
    def __init__(self):
        self.nodes = set()
        self.edges = []
        self.optimizations = {
            'constant_folding': True,
            'common_subexpression_elimination': True,
            'dead_code_elimination': True
        }

    def build_graph(self, output_node):
        """
              构建计算图。

              Args:
                  output_node (Variable): 输出节点。

              Returns:
                  ComputationGraph: 返回自身。
              """
        def add_node_and_edges(node):
            self.nodes.add(node)
            if isinstance(node, Variable) and node.creator is not None:
                func = node.creator
                add_node_and_edges(func)
                for input_node in func.inputs:
                    self.edges.append((input_node, func))
                    self.edges.append((func, node))
                    add_node_and_edges(input_node)

        add_node_and_edges(output_node)
        return self

    def visualize(self, filename='computation_graph'):
        dot = Digraph(comment='Computation Graph')
        dot.attr(rankdir='LR')

        for node in self.nodes:
            if isinstance(node, Variable):
                label = f"{node.name if node.name else 'Variable'}\n"
                label += f"data: {node.data.shape if node.data.size > 2 else node.data}"
                if node.grad is not None:
                    label += f"\ngrad: {node.grad.data.shape if node.grad.data.size > 2 else node.grad.data}"
                shape = 'ellipse'
            else:
                label = node.__class__.__name__
                shape = 'box'

            dot.node(str(id(node)), label, shape=shape)

        for edge in self.edges:
            dot.edge(str(id(edge[0])), str(id(edge[1])))

        dot.render(filename, format='png', cleanup=True, view=True)
        print(f"计算图已保存为 {filename}.png")

    def create_dot(self):
        dot = Digraph(comment='Computation Graph')
        dot.attr(rankdir='LR')

        for node in self.nodes:
            if isinstance(node, Variable):
                label = f"{node.name if node.name else 'Variable'}\n"
                if node.data is not None:
                    label += f"data: {node.data.shape if node.data.size > 2 else node.data}"
                if node.grad is not None:
                    label += f"\ngrad: {node.grad.data.shape if node.grad.data.size > 2 else node.grad.data}"
                shape = 'ellipse'
            else:
                label = node.__class__.__name__
                shape = 'box'

            dot.node(str(id(node)), label, shape=shape)

        for edge in self.edges:
            dot.edge(str(id(edge[0])), str(id(edge[1])))

        return dot

    def optimize(self):
        print("执行图优化...")
        if self.optimizations['constant_folding']:
            self.constant_folding()
        if self.optimizations['common_subexpression_elimination']:
            self.common_subexpression_elimination()
        if self.optimizations['dead_code_elimination']:
            self.dead_code_elimination()

    def constant_folding(self):
        print("执行常量折叠...")
        changed = True
        while changed:
            changed = False
            for node in list(self.nodes):
                if isinstance(node, Variable) and node.creator is not None:
                    func = node.creator
                    if all(isinstance(input_node, Variable) and input_node.creator is None for input_node in
                           func.inputs):
                        with contextlib.nullcontext():
                            result = func.forward(*[input_node.data for input_node in func.inputs])
                        node.data = result
                        node.creator = None
                        self.nodes.remove(func)
                        self.edges = [edge for edge in self.edges if edge[1] != func and edge[0] != func]
                        changed = True

    def common_subexpression_elimination(self):
        print("执行公共子表达式消除...")
        expression_map = {}
        for node in list(self.nodes):
            if isinstance(node, Variable) and node.creator is not None:
                func = node.creator
                key = (func.__class__, tuple(id(input_node) for input_node in func.inputs))
                if key in expression_map:
                    existing_node = expression_map[key]
                    node.data = existing_node.data
                    node.creator = None
                    self.nodes.remove(func)
                    self.edges = [edge for edge in self.edges if edge[1] != func and edge[0] != func]
                    self.edges.append((existing_node, node))
                else:
                    expression_map[key] = node

    def dead_code_elimination(self):
        print("执行死代码消除...")
        used_nodes = set()
        output_nodes = [node for node in self.nodes if isinstance(node, Variable) and node.grad is not None]

        def mark_used(node):
            if node not in used_nodes:
                used_nodes.add(node)
                if isinstance(node, Variable) and node.creator is not None:
                    mark_used(node.creator)
                    for input_node in node.creator.inputs:
                        mark_used(input_node)

        for output_node in output_nodes:
            mark_used(output_node)

        self.nodes = used_nodes
        self.edges = [(f, t) for f, t in self.edges if f in used_nodes and t in used_nodes]

# =============================================================================
# Variable / Function
# =============================================================================
try:
    import torch
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray)


class Variable:
    """
       变量类，表示计算图中的节点。

       Attributes:
           data: 变量的数据。
           name (str): 变量的名称。
           grad: 变量的梯度。
           creator (Function): 创建该变量的函数。
           generation (int): 变量的世代。
       """

    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def unchain(self):
        self.creator = None

    def cleargrad(self):
        self.grad = None

    def build_and_visualize_graph(self, window_name='Computation Graph'):
        """
         构建并可视化当前变量的计算图, 用于前向传播时想看看计算图的时候

         Args:
             filename (str): 保存可视化图的文件名。
         """
        graph = ComputationGraph()
        graph.build_graph(self)
        dot = graph.create_dot()
        img = graphviz_to_opencv(dot)

        if not Config.cv_window_created:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            Config.cv_window_created = True

        cv2.imshow(window_name, img)
        cv2.waitKey(1)  # Wait for a short time to update the window
        return img

    def backward(self, retain_grad=True, optimize=False, visualize=False):
        """
            执行反向传播。

            Args:
                retain_grad (bool): 是否保留中间变量的梯度。
                optimize (bool): 是否优化计算图。
                visualize (bool): 是否可视化计算图。

            """
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        graph = ComputationGraph()
        graph.build_graph(self)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        input_variables = set()
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)
                else:
                    input_variables.add(x)  # This is an input variable

        if optimize:
            graph.optimize()

        if visualize:
            graph.visualize()

        if not retain_grad:
            for node in graph.nodes:
                if isinstance(node, Variable):
                    if node is not self and node not in input_variables:
                        node.grad = None

    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return samsara.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return samsara.functions.transpose(self, axes)

    @property
    def T(self):
        return samsara.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return samsara.functions.sum(self, axis, keepdims)

    def to_cpu(self):
        if self.data is not None:
            self.data = samsara.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = samsara.cuda.as_cupy(self.data)


class Parameter(Variable):
    pass


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


class Function:
    """
      函数类，表示计算图中的操作。

      Attributes:
          inputs (list): 输入变量列表。
          outputs (list): 输出变量列表。
          generation (int): 函数的世代。
      """
    def __call__(self, *inputs):
        """
              调用函数，执行前向传播。

              Args:
                  *inputs: 输入变量。

              Returns:
                  Variable or list of Variables: 函数的输出。
              """
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        # 在这里添加可视化选项
        if Config.visualize_forward:
            for output in outputs:
                output.build_and_visualize_graph()

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        """
               前向传播的具体实现。

               Args:
                   xs (list): 输入数据列表。

               Returns:
                   Variable or list of Variables: 函数的输出。
               """
        raise NotImplementedError()

    def backward(self, gys):
        """
               反向传播的具体实现。

               Args:
                   gys (list): 输出梯度列表。

               Returns:
                   Variable or list of Variables: 输入梯度。
               """
        raise NotImplementedError()


class FusedFunction(Function):
    """
      融合函数类，用于将多个函数合并为一个操作, 未完成

      Attributes:
          outer_func (Function): 外部函数。
          inner_funcs (tuple of Function): 内部函数列表。
      """
    def __init__(self, outer_func, *inner_funcs):
        self.outer_func = outer_func
        self.inner_funcs = inner_funcs

    def forward(self, *xs):
        inner_ys = [f.forward(*xs[i:i+len(f.inputs)]) for i, f in enumerate(self.inner_funcs)]
        return self.outer_func.forward(*inner_ys)

    def backward(self, gy):
        raise NotImplementedError("Backward pass for fused functions is not implemented")


# =============================================================================
# Utility functions (新添加)
# =============================================================================
def visualize_graph(output_variable: Variable, filename='computation_graph'):
    graph = ComputationGraph()
    graph.build_graph(output_variable)
    graph.visualize(filename)


def optimize_graph(output_variable: Variable):
    graph = ComputationGraph()
    graph.build_graph(output_variable)
    graph.optimize()


# =============================================================================
# 运算符重载
# =============================================================================
class Add(Function):
    def forward(self, x0, x1):
        """
               执行加法的前向传播。

               Args:
                   x0, x1: 输入数据。

               Returns:
                   加法结果。
               """
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        """
               执行加法的反向传播。

               Args:
                   gy: 输出梯度。

               Returns:
                   输入梯度元组。
               """
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:  # for broadcaset
            gx0 = samsara.functions.sum_to(gx0, self.x0_shape)
            gx1 = samsara.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    """
       加法函数。

       Args:
           x0, x1: 输入变量。

       Returns:
           加法结果变量。
       """
    x1 = as_array(x1, samsara.cuda.get_array_module(x0.data))
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = samsara.functions.sum_to(gx0, x0.shape)
            gx1 = samsara.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1):
    x1 = as_array(x1, samsara.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = samsara.functions.sum_to(gx0, self.x0_shape)
            gx1 = samsara.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1, samsara.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1, samsara.cuda.get_array_module(x0.data))
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = samsara.functions.sum_to(gx0, x0.shape)
            gx1 = samsara.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1, samsara.cuda.get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1, samsara.cuda.get_array_module(x0.data))
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    """
        设置变量类的运算符重载。
    """
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = samsara.functions.get_item

    Variable.matmul = samsara.functions.matmul
    Variable.dot = samsara.functions.matmul
    Variable.max = samsara.functions.max
    Variable.min = samsara.functions.min


# 测试代码
def test_computation_graph():
    setup_variable()
    print("开始测试计算图功能...")

    def goldstein(x, y):
        z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
            (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
        return z

    x = Variable(np.array(1.0), name='x')
    y = Variable(np.array(1.0), name='y')
    # 使用上下文管理器来启用前向传播的可视化
    with using_visualize_forward():
        z = goldstein(x, y)
    z.name = 'z'

    print(f"计算结果: v = {z.data}")

    print("\n测试反向传播、优化和可视化...")

    z.backward(optimize=False, visualize=False, retain_grad=False)

    print(f"\n梯度: dx = {x.grad.data}, dy = {y.grad.data}")

    print("\n测试完成！按任意键关闭窗口。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_computation_graph()