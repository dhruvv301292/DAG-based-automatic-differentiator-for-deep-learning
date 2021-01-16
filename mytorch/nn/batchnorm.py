from mytorch.tensor import Tensor
import numpy as np
from mytorch.nn.module import Module
import torch
import torch.nn as nn

class BatchNorm1d(Module):
    """Batch Normalization Layer

    Args:
        num_features (int): # dims in input and output
        eps (float): value added to denominator for numerical stability
                     (not important for now)
        momentum (float): value used for running mean and var computation (alpha)

    Inherits from:
        Module (mytorch.nn.module.Module)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features

        self.eps = Tensor(np.array([eps]))
        self.momentum = Tensor(np.array([momentum]))
        self.one_minus_momentum = Tensor(np.array([1.0 - momentum]))

        # To make the final output affine
        self.gamma = Tensor(np.ones((self.num_features,)), requires_grad=True, is_parameter=True)
        self.beta = Tensor(np.zeros((self.num_features,)), requires_grad=True, is_parameter=True)

        # Running mean and var
        self.running_mean = Tensor(np.zeros(self.num_features,), requires_grad=False, is_parameter=False)
        self.running_var = Tensor(np.ones(self.num_features,), requires_grad=False, is_parameter=False)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, num_features)
        Returns:
            Tensor: (batch_size, num_features)
        """
        batch_size = Tensor(np.array([x.shape[0]]), requires_grad=False, is_parameter=False)
        batch_size_unbiased = Tensor(np.array([x.shape[0]-1]), requires_grad=False, is_parameter=False)

        if self.is_train:
            mean = x.sum ( axis=0 ) / batch_size

            variance = (x - mean).square().sum(axis=0) / batch_size
            variance_unbiased = (x - mean).square().sum(axis = 0) / batch_size_unbiased  #######

            self.running_mean = (self.one_minus_momentum * self.running_mean) + (self.momentum * mean)  #######
            self.running_var = (self.one_minus_momentum * self.running_var) + (self.momentum * variance_unbiased)  #######

            x_norm_train = (x - mean) /(variance + self.eps).sqrt()
            return (self.gamma * x_norm_train) + self.beta

        else:
            x_norm_eval = (x - self.running_mean) / (self.running_var + self.eps).sqrt()  #######
            return (self.gamma * x_norm_eval) + self.beta  #######


if __name__ == '__main__':
    a = Tensor(np.array([[1.,2.], [3.,4.]]))
    my_bn = BatchNorm1d(a.shape[1])


    a_py = torch.FloatTensor([[1,2],[3,4]])
    py_bn = nn.BatchNorm1d(a_py.shape[1])

    my_bn.is_train = True
    py_bn.train(True)

    py_y = py_bn(a_py)
    my_y = my_bn.forward(a)
    print()
    my_y.backward()
    py_y.sum().backward()