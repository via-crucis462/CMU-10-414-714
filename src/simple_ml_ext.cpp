#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
    * softmax 回归 epoch 函数的 C++ 版本。该函数应使用 X 和 y 定义的数据
    * （以及 m、n、k 指定的尺寸）运行一个 epoch，并原地修改 theta。你可能
    * 需要分配（并随后释放）一些辅助数组，用于存储 logits 和梯度。
     *
    * 参数：
    *     X (const float *): 指向 X 数据的指针，大小为 m*n，按行主序
    *          （C 风格）存储
    *     y (const unsigned char *): 指向 y 数据的指针，大小为 m
    *     theta (float *): 指向 theta 数据的指针，大小为 n*k，按行主序
    *          （C 风格）存储
    *     m (size_t): 样本数量
    *     n (size_t): 输入维度
    *     k (size_t): 类别数量
    *     lr (float): 学习率 / SGD 步长
    *     batch (int): SGD 小批量大小
     *
    * 返回值：
    *     （无）
     */

    /// BEGIN YOUR CODE
    for(int i = 0 ; i < m ; i += batch){
        int current_batch_size = std::min(batch, m - i);
        const float *X_batch = X + i * n;
        const unsigned char *y_batch = y + i;

        // 前向传播

        // h(x) per batch
        float *logits = new float[current_batch_size * k];
        for(int j = 0 ; j < current_batch_size ; ++j){
            for(int c = 0 ; c < k ; ++c){
                logits[j * k + c] = 0.0f;
                for(int d = 0 ; d < n ; ++d){
                    logits[j * k + c] += X_batch[j * n + d] * theta[d * k + c];
                }
            }
        }

        // 梯度计算：先累积整个 batch 的梯度，再统一更新参数
        float *gradient = new float[n * k]();
        for(int j = 0 ; j < current_batch_size ; ++j){
            int y_true = y_batch[j];
            float prob = 0.0f;
            for(int c = 0 ; c < k ; ++c){
                prob += std::exp(logits[j * k + c]);
            }
            for(int c = 0 ; c < k ; ++c){
                float softmax = std::exp(logits[j * k + c]) / prob;
                float grad = (softmax - (c == y_true ? 1.0f : 0.0f)) / current_batch_size;
                for(int d = 0 ; d < n ; ++d){
                    gradient[d * k + c] += grad * X_batch[j * n + d];
                }
            }
        }

        for(int d = 0 ; d < n ; ++d){
            for(int c = 0 ; c < k ; ++c){
                theta[d * k + c] -= lr * gradient[d * k + c];
            }
        }

        delete[] gradient;
        delete[] logits;
    }
    /// END YOUR CODE
}


/**
 * 以下是用于封装上述函数的 pybind11 代码。它唯一的作用是将上述函数
 * 封装到 Python 模块中，无需修改这部分代码。
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
