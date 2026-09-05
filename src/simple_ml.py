import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ 一个简单的 'add' 函数，供你熟悉自动评分与提交系统。本题的解答
    在作业 notebook 中。

    参数:
        x (Python 数字或 numpy 数组)
        y (Python 数字或 numpy 数组)

    返回:
        x + y 的和
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ 读取 MNIST 格式的图像和标签文件。参见该网页了解文件格式的说明：
    http://yann.lecun.com/exdb/mnist/

    参数:
        image_filename (str): MNIST 格式的 gzip 压缩图像文件名
        label_filename (str): MNIST 格式的 gzip 压缩标签文件名

    返回:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 包含所加载数据的二维 numpy 数组，
                其维度应为 (num_examples x input_dim)，其中 'input_dim' 是
                数据的完整维度，例如 MNIST 图像为 28x28，因此为 784。
                数值应为 np.float32 类型，且数据应归一化到最小值为 0.0、
                最大值为 1.0（即把原始值 0 缩放到 0.0、255 缩放到 1.0）。

            y (numpy.ndarray[dtype=np.uint8]): 包含各示例标签的一维 numpy
                数组。数值应为 np.uint8 类型，MNIST 中取值为 0-9。
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        image_data = np.frombuffer(f.read(), dtype=np.uint8)

    with gzip.open(label_filename, 'rb') as f:
        magic_y, num_labels = struct.unpack('>II', f.read(8))
        label_data = np.frombuffer(f.read(), dtype=np.uint8)

    if num_images != num_labels:
        raise ValueError("Image count and label count do not match")

    X = image_data.reshape(num_images, rows * cols).astype(np.float32) / 255.0
    y = label_data.astype(np.uint8)
    return X, y
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ 返回 softmax 损失。注意：就本作业而言，你无需担心对 log-sum-exp
    计算的数值属性做"优雅"的缩放处理，直接计算即可。

    参数:
        Z (np.ndarray[np.float32]): 形状为 (batch_size, num_classes) 的
            二维 numpy 数组，包含每个类别的 logit 预测值。
        y (np.ndarray[np.uint8]): 形状为 (batch_size, ) 的一维 numpy 数组，
            包含每个样本的真实标签。

    返回:
        所有样本的平均 softmax 损失。
    """
    ### BEGIN YOUR CODE
    sum = 0.0
    for i in range(Z.shape[0]):
        xi = Z[i]
        yi = y[i]
        hy = xi[yi]
        log_sum_exp = np.log(np.sum(np.exp(xi)))
        loss_i = -hy + log_sum_exp
        sum += loss_i
    return sum / Z.shape[0]
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ 在数据上运行一轮 softmax 回归的 SGD，使用步长 lr 和指定的 batch
    大小。该函数应就地修改 theta 矩阵，并且应_不_打乱顺序地遍历 X 中的
    批次。

    参数:
        X (np.ndarray[np.float32]): 大小为 (num_examples x input_dim) 的
            二维输入数组。
        y (np.ndarray[np.uint8]): 大小为 (num_examples,) 的一维类别标签数组
        theta (np.ndarrray[np.float32]): softmax 回归参数的二维数组，
            形状为 (input_dim, num_classes)
        lr (float): SGD 的步长（学习率）
        batch (int): SGD 小批量的大小

    返回:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    for start in range(0, num_examples, batch):
        end = min(start + batch, num_examples)
        X_batch = X[start:end]
        y_batch = y[start:end]

        logits = X_batch @ theta
        exp_logits = np.exp(logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True) #axis表示沿哪个维度进行操作，1表示第二个维度，即列

        one_hot_y = np.zeros_like(softmax_probs) # 创造一个与softmax_probs形状相同的全零数组
        one_hot_y[np.arange(len(y_batch)), y_batch] = 1 

        # one_hot_y[[0, 1, 2], [2, 0, 1]] = 1 高级索引
        # 两两配对

        gradient = X_batch.T @ (softmax_probs - one_hot_y) / len(y_batch)
        theta -= lr * gradient
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ 针对由权重 W1 和 W2 定义（无偏置项）的两层神经网络运行一轮 SGD：
        logits = ReLU(X * W1) * W2
    该函数应使用步长 lr 和指定的 batch 大小（同样不打乱 X 的顺序），并且
    应就地修改 W1 和 W2 矩阵。

    参数:
        X (np.ndarray[np.float32]): 大小为 (num_examples x input_dim) 的
            二维输入数组。
        y (np.ndarray[np.uint8]): 大小为 (num_examples,) 的一维类别标签数组
        W1 (np.ndarray[np.float32]): 第一层权重的二维数组，形状为
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 第二层权重的二维数组，形状为
            (hidden_dim, num_classes)
        lr (float): SGD 的步长（学习率）
        batch (int): SGD 小批量的大小

    返回:
        None
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE



### 以下代码仅供演示，你无需编辑

def loss_err(h,y):
    """ 计算损失和误差的辅助函数"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ 用于完整训练 softmax 回归分类器的示例函数 """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ 用于训练两层神经网络的示例函数 """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
