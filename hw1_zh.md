# 10-714：作业 1

本作业将帮助你开始实现本课程中会持续开发的 **needle**（**ne**cessary **e**lements of **d**eep **le**arning，深度学习的必要元素）库。具体来说，本作业的目标是构建一个基础的**自动微分**框架，然后使用它重新实现你在 HW0 中用于 MNIST 手写数字分类的简单两层神经网络。

首先，像 HW0 一样，通过“File”菜单中的“Save a copy in Drive”复制此 notebook 文件，然后运行下面的代码单元格。接着运行用于配置并安装必要软件包的代码。

```python
# 配置作业环境
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/
!mkdir -p 10714
%cd /content/drive/MyDrive/10714
!git clone https://github.com/dlsys10714/hw1.git
%cd /content/drive/MyDrive/10714/hw1

!pip3 install --upgrade --no-deps git+https://github.com/dlsys10714/mugrade.git
!pip3 install numdifftools
```

```python
import sys
sys.path.append('./python')
sys.path.append('./apps')
from simple_ml import *
```

## `needle` 简介

如需了解 `needle` 框架，请参考课堂 Lecture 5，以及配套的[这份 Jupyter notebook](https://github.com/dlsys10714/notebooks/blob/main/5_automatic_differentiation_implementation.ipynb)。在本作业中，你将使用 `numpy` CPU 后端实现自动微分的基础功能；在后续作业中，你会迁移到自己的线性代数库，并加入 GPU 代码。本作业的所有代码都使用 Python 编写。

对于本作业，`needle` 库中有两个重要文件：`python/needle/autograd.py` 定义计算图的基础结构，并将成为自动微分框架的基础；`python/needle/ops.py` 包含各种运算符的实现，你将在本作业以及后续课程中使用并完善这些运算符。

虽然 `autograd.py` 中的自动微分基础框架已经搭建完成，但你仍应熟悉库中几个类的基本概念。我们**不建议**你在开始实现之前就通读整个代码库，因为有些功能在完成部分实现后会更容易理解。不过，你应该先了解 `needle` 中类的基本结构和组织方式，尤其是以下几个类：

- `Value`：计算图中的一个值，可以是对其他 `Value` 对象执行运算后的输出，也可以是常量（叶节点）`Value`。这里使用通用类，以便后续支持其他数据结构；但目前你主要会通过它的子类 `Tensor` 与之交互。
- `Op`：计算图中的一个运算符。运算符需要在 `compute()` 方法中定义前向传播，即如何对 `Value` 对象的底层数据执行运算；还需要通过 `gradient()` 方法定义反向传播，即如何与传入的输出梯度相乘。具体实现细节将在下面说明。
- `Tensor`：`Value` 的子类，对应计算图中的实际张量输出，即一个多维数组。本作业以及之后大多数作业都会使用这个类，而不是上面的通用 `Value` 类。我们提供了一些便捷功能（例如运算符重载），让你可以使用普通 Python 语法操作张量，但这些功能要到对应运算实现后才能正常工作。
- `TensorOp`：用于返回 `Tensor` 的 `Op` 子类。本作业中需要实现的所有运算都属于这一类型。

## 问题 1：实现前向计算 [10 分]

首先，你需要为新的运算符实现前向计算。为了说明实现方式，下面以 `ops.py` 中的 `EWiseAdd` 运算符为例：

```python
class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad

def add(a, b):
    return EWiseAdd()(a, b)
```

这类实现遵循以下约定。`compute()` 函数执行“前向”计算，也就是直接完成相应运算。但需要特别注意：`compute` 的输入都是 `NDArray` 对象（在当前实现中，它们实际上是 `numpy.ndarray`；后续作业中你会实现自己的 `NDArray`）。也就是说，`compute()` 针对的是原始数据对象，而不是自动微分计算图中的 `Tensor` 对象。

下一节会介绍 `gradient()` 调用，但这里需要强调，它与前向计算不同，因为它接收的是 `Tensor` 参数。因此，在这个函数中进行的任何调用都应该通过 `TensorOp` 运算本身完成，这样才能对梯度继续求梯度。

此外，我们还定义了辅助函数 `add()`，这样就不必使用较为繁琐的 `EWiseAdd()(a, b)` 来相加两个 `Tensor`。这些函数都已经为你写好，应该不难理解。

本题要求你为下面每个类实现 `compute` 调用。这些调用非常直接，通常只需要调用相关的 NumPy 函数即可。由于后续作业中会使用非 NumPy 后端，我们将 NumPy 导入为 `import numpy as array_api`；如果你想使用常见的 `np.X()` 调用，应改用 `array_api.add()` 等形式。

- `PowerScalar`：将输入提升到整数（标量）次幂
- `EWiseDiv`：对两个输入逐元素执行真除法
- `DivScalar`：将输入逐元素除以一个标量（一个输入，`scalar` 为数字）
- `MatMul`：对两个输入执行矩阵乘法
- `Summation`：沿给定轴对数组元素求和（一个输入，`axes` 为元组）
- `BroadcastTo`：将数组广播到新的形状（一个输入，`shape` 为元组）
- `Reshape`：在不改变数据的情况下改变数组形状（一个输入，`shape` 为元组）
- `Negate`：逐元素取数值负值（一个输入）
- `Transpose`：交换两个轴；默认交换最后两个轴（一个输入，`axes` 为元组）

```bash
python3 -m pytest -v -k "forward"
```

```bash
python3 -m mugrade submit 'YOUR_GRADER_KEY_HERE' -k "forward"
```

## 问题 2：实现反向计算 [25 分]

现在你已经实现了计算图中的一些函数。为了实现自动微分，我们需要执行反向传播，也就是将函数的相关导数与传入的反向梯度相乘。

实现这些计算的一种简单方法仍然是求“假设所有量都是标量时”的偏导数，然后匹配张量尺寸。我们提供的测试会自动将你的结果与数值导数进行比较，以检查实现是否正确。

反向模式自动微分的总体目标，是计算某个下游函数 $\ell$（它依赖于 $f(x,y)$）关于 $x$（或 $y$）的梯度。形式化地说，我们希望计算：

$$
\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial x}.
$$

“传入的反向梯度”正是 $\frac{\partial \ell}{\partial f(x,y)}$，因此 `gradient()` 函数最终需要计算该反向梯度与函数自身关于 $x$ 的导数 $\frac{\partial f(x,y)}{\partial x}$ 的乘积。

更具体地看，考虑前面介绍过的逐元素加法：

$$
f(x,y) = x + y.
$$

假设 $x,y\in\mathbb{R}^n$，因此 $f(x,y) \in \mathbb{R}^n$。通过简单求导：

$$
\frac{\partial f(x,y)}{\partial x} = 1
$$

于是：

$$
\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial x} = \frac{\partial \ell}{\partial f(x,y)}
$$

也就是说，关于第一个参数 $x$ 的导数与传入的反向梯度完全相同。关于第二个参数 $y$ 也是如此。这正是 `EWiseAdd` 运算符中以下方法所表达的含义：

```python
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad
```

该函数直接返回传入的反向梯度（这里它实际上就是传入的反向梯度与关于每个参数的导数的乘积）。由于 $f(x,y)$、$x$ 和 $y$ 的尺寸相同，因此无需额外处理维度。

再考虑逐元素乘法：

$$
f(x,y) = x \circ y
$$

其中 $\circ$ 表示逐元素乘法。该函数的偏导数为：

$$
\frac{\partial f(x,y)}{\partial x} = y
$$

以及：

$$
\frac{\partial f(x,y)}{\partial y} = x
$$

因此，关于 $x$ 的梯度为：

$$
\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial x} = \frac{\partial \ell}{\partial f(x,y)} \cdot y
$$

如果和上一个例子一样，$x,y \in \mathbb{R}^n$，那么 $f(x,y) \in \mathbb{R}^n$，因此梯度函数返回的第一个元素就是逐元素乘法：

$$
\frac{\partial \ell}{\partial f(x,y)} \circ y
$$

这正是 `EWiseMul` 类的 `gradient()` 调用所表达的含义：

```python
class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs
```

### 实现反向传播

与前向传播函数不同，`gradient` 函数的参数是 `needle` 对象。必须只使用 `needle` 运算（即 `python/needle/ops.py` 中定义的运算）来实现反向传播，而不是对底层 NumPy 数据调用 NumPy 运算，这样才能将梯度本身构建为计算图。唯一的例外是下面定义的 `ReLU` 运算：由于其梯度本身不可微，可以直接访问 `Tensor` 中的数据，但这属于特殊情况。

本题要求为以下类填写 `gradient` 函数：

- `EWiseDiv`
- `DivScalar`
- `MatMul`
- `Summation`
- `BroadcastTo`
- `Reshape`
- `Negate`
- `Transpose`

所有这些梯度函数都可以只使用 `python/needle/ops.py` 中已有的运算实现，因此不需要定义其他前向函数。

**提示：**乘法、除法等运算的梯度通常比较直观，但 `Broadcast` 或 `Summation` 等运算的反向传播可能不那么容易理解。你可以进行数值梯度检查并打印实际梯度值来获得线索；参考 `tests/test_autograd_hw.py` 中的 `check_gradients()` 函数。请记住，`out_grad` 的尺寸始终等于运算输出的尺寸，而 `gradient()` 返回的 `Tensor` 尺寸必须始终与运算原始输入的尺寸相同。

### 检查反向传播

再次强调，可以通过数值梯度检查验证反向传播是否正确。正如课堂中介绍的：

$$
\delta^T \nabla_\theta f(\theta) = \frac{f(\theta + \epsilon \delta) - f(\theta - \epsilon \delta)}{2 \epsilon} + o(\epsilon^2)
$$

我们在 `tests/test_autograd.py` 中提供了用于数值检查的 `gradient_check` 函数。

```bash
python3 -m pytest -l -v -k "backward"
```

```bash
python3 -m mugrade submit 'YOUR_GRADER_KEY_HERE' -k "backward"
```

## 问题 3：拓扑排序 [20 分]

现在，你的系统已经可以对张量执行运算并构建计算图。接下来，你需要编写自动微分所需的一个核心工具： [拓扑排序](https://en.wikipedia.org/wiki/Topological_sorting)。它将允许我们遍历计算图（进行前向或反向遍历），并在遍历过程中计算梯度。此外，已有组件还允许我们在反向拓扑遍历期间执行运算，从而继续向计算图中添加节点（正如课堂中所讨论的），并且可以“免费”获得高阶导数。

请在 `python/needle/autograd.py` 中完成 `find_topo_sort` 方法和 `topo_sort_dfs` 辅助方法，实现拓扑排序。

#### 提示

- 必须执行后序深度优先搜索，否则测试会失败。
- `topo_sort_dfs` 方法不是必需的，但我们发现使用递归辅助函数很方便。
- Lecture 4 幻灯片中“通过扩展计算图实现反向模式自动微分”一节展示了正确节点顺序的示例。
- 后续部分会反向遍历这个排序结果，但 `find_topo_sort` 应该返回正向的节点顺序。

```bash
python3 -m pytest -k "topo_sort"
```

```bash
python3 -m mugrade submit 'YOUR_GRADER_KEY_HERE' -k "topo_sort"
```

## 问题 4：实现反向模式微分 [25 分]

正确实现拓扑排序后，接下来就可以利用它实现反向模式自动微分算法。回顾上一节课的内容，我们需要按照逆拓扑顺序遍历计算图，并构建新的伴随节点。请在 `python/needle/autograd.py` 的 `compute_gradient_of_variables` 函数中实现 Reverse AD 算法。这将启用 `Tensor` 类的 `backward` 方法，该方法负责计算梯度并将其存储在每个输入 `Tensor` 的 `grad` 字段中。完成后，我们的反向模式自动微分引擎就可以工作了。

我们可以采用与单独检查反向梯度类似的方法验证实现正确性：使用 `tests/test_autograd.py` 中的 `gradient_check` 函数，将数值梯度与计算得到的梯度进行比较。

> 原 notebook 在此处包含一张课堂示意图。为避免在 Markdown 中嵌入约 138 KB 的 base64 图片数据，这里保留图片位置说明。

> 此处原 notebook 包含一张反向模式自动微分示意图。

正如课堂中所讨论的，反向模式自动微分的结果仍然是一个计算图。我们可以通过组合更多运算来继续扩展这个计算图，然后再次对梯度执行反向模式自动微分（本题最后两个测试会检查这一点）。

```bash
python3 -m pytest -k "compute_gradient"
```

```bash
python3 -m mugrade submit 'YOUR_GRADER_KEY_HERE' -k "compute_gradient"
```

## 问题 5：Softmax 损失 [10 分]

下面的问题将使用 MNIST 数据集进行测试，因此我们会使用 HW0 中编写的 `parse_mnist` 函数。

1. 首先，将 HW0 问题 2 的解决方案复制到 `apps/simple_ml.py` 文件中的 `parse_mnist` 函数。

本题要求实现 `apps/simple_ml.py` 中定义的 `softmax_loss()` 函数。它与 HW0 问题 3 中的 softmax 损失类似，但这次输入的是 logits 的 `Tensor` 和真实标签 one-hot 编码的 `Tensor`。回顾一下，对于取值为 $y \in \{1,\ldots,k\}$ 的多分类输出，softmax 损失接收 logits 向量 $z \in \mathbb{R}^k$、真实类别 $y \in \{1,\ldots,k\}$（在本函数中以 one-hot 向量表示），并返回：

$$
\ell_{\mathrm{softmax}}(z, y) = \log\sum_{i=1}^k \exp z_i - z_y.
$$

首先，你需要实现另一个运算符 `log` 的前向和反向传播。

2. 在 `python/needle/ops.py` 的 `Log` 运算符中填写 `compute()` 函数。
3. 在 `python/needle/ops.py` 的 `Log` 运算符中填写 `gradient()` 函数。

完成这些运算符后：

4. 在 `apps/simple_ml.py` 中实现 `softmax_loss` 函数。

你可以从 HW0 的解决方案开始，然后修改为兼容 `needle` 对象和运算。和之前的作业一样，该函数应该计算大小为 $m$ 的 batch 上的**平均** softmax 损失：logits `Z` 是一个 $m \times k$ 的 `Tensor`，每一行表示一个样本；`y_one_hot` 是一个 $m \times k$ 的 `Tensor`，除了每行真实标签对应位置为 1 外，其余位置均为 0。最后，请注意，返回的平均 softmax 损失也应该是一个 `Tensor`。

```bash
python3 -m pytest -k "softmax_loss_ndl"
```

```bash
python3 -m mugrade submit 'YOUR_GRADER_KEY_HERE' -k "softmax_loss_ndl"
```

## 问题 6：两层神经网络的 SGD [10 分]

和 HW0 一样，现在需要为一个简单的两层神经网络实现随机梯度下降（SGD），网络定义见 HW0 问题 5。

具体来说，对于输入 $x \in \mathbb{R}^n$，考虑如下不带偏置的两层神经网络：

$$
z = W_2^T \mathrm{ReLU}(W_1^T x)
$$

其中 $W_1 \in \mathbb{R}^{n \times d}$ 和 $W_2 \in \mathbb{R}^{d \times k}$ 是网络权重（网络具有 $d$ 维隐藏单元），$z \in \mathbb{R}^k$ 是网络输出的 logits。我们仍然使用 softmax / 交叉熵损失，因此希望求解以下优化问题。这里将记号扩展到 batch 形式，输入矩阵为 $X \in \mathbb{R}^{m \times n}$：

$$
\min_{W_1, W_2} \;\; \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y).
$$

首先，你需要实现 `relu` 运算符的前向和反向传播。

1. 在 `python/needle/ops.py` 中填写 `ReLU` 运算符的函数。
2. 在 `python/needle/ops.py` 中填写 `ReLU` 类的 `gradient` 函数。**注意：**在这个特定情况下，可以访问输出 Tensor 的 `.realize_cached_data()`，因为 ReLU 不是二阶可微的。

然后：

3. 在 `apps/simple_ml.py` 中填写 `nn_epoch` 方法。

同样，你可以使用 HW0 中的 `nn_epoch` 函数作为起点。注意，与 HW0 不同，这里的输入 `W1` 和 `W2` 是 `Tensor`。但输入 `X` 和 `y` 仍然是 NumPy 数组，因此你需要像 HW0 一样遍历 `X` 和 `y` 的 mini-batch，然后将每个 `X_batch` 转换为 `Tensor`，并将 `y_batch` 进行 one-hot 编码后转换为 `Tensor`。上次你直接推导了这个两层 ReLU 网络的反向传播方程，而这次将通过调用 `Tensor` 类的 `.backward()` 方法，使用自动微分引擎自动计算梯度。对于每个 mini-batch，调用 `.backward()` 后，使用 NumPy 计算更新后的 `W1` 和 `W2`，然后使用这些 NumPy 数值创建新的 `W1` 和 `W2` `Tensor`。你的实现应返回最终的 `W1` 和 `W2` `Tensor`。

```bash
python3 -m pytest -l -k "nn_epoch_ndl"
```

```bash
python3 -m mugrade submit 'YOUR_GRADER_KEY_HERE' -k "nn_epoch_ndl"
```
