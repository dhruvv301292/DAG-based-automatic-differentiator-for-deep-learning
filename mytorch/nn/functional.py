import numpy as np

from mytorch import tensor
from mytorch.autograd_engine import Function


def checkBroadcast(a, b):
    min_dim = min ( len ( a.shape ), len ( b.shape ) )
    for i in range(-1, -min_dim - 1, -1):
        ashape = a.shape[i]
        bshape = b.shape[i]
        if (ashape != bshape) and (ashape != 1 and bshape != 1):
            return False
    return True

def unbroadcast(a, grad_a): #input and grad are ndarrays
    a_shape = list(a.shape)
    grad_shape = list(grad_a.shape)
    if len(grad_shape) > len(a_shape):
        a_shape[0:0] = [0]*(len(grad_shape)-len(a_shape))
    grad_a_unbroadcasted = grad_a
    for i in range(len(a_shape)):
        if a_shape[i] != grad_shape[i]:
            grad_a_unbroadcasted = grad_a_unbroadcasted.sum(axis = i, keepdims = True)
    grad_a_unbroadcasted = grad_a_unbroadcasted.reshape(a.shape)
    return grad_a_unbroadcasted

class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                                    is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.T)

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                                                 is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data)

"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""
class Add(Function):

    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape
        if (a.data.shape != b.data.shape) and not checkBroadcast(a.data, b.data):
            raise Exception("Both args should either have same sizes or be broadcastable: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad, is_leaf=not requires_grad)
        return c


    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        if a.shape == grad_output.shape:
            grad_a = np.ones(a.shape) * grad_output.data
        else:
            grad_a = unbroadcast(a.data, grad_output.data)

        if b.shape == grad_output.shape:
            grad_b = np.ones(b.shape) * grad_output.data
        else:
            grad_b = unbroadcast(b.data, grad_output.data)

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # check the inputs are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # check the inputs are of same shape
        if (a.data.shape != b.data.shape) and not checkBroadcast ( a.data, b.data ):
            raise Exception ("Both args should either have same sizes or be broadcastable: {}, {}".format(a.shape, b.shape))

        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor ( a.data - b.data, requires_grad=requires_grad, is_leaf=not requires_grad )
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        if a.shape == grad_output.shape:
            grad_a = np.ones(a.shape) * grad_output.data
        else:
            grad_a = unbroadcast(a.data, grad_output.data)

        if b.shape == grad_output.shape:
            grad_b = -1 * np.ones(b.shape) * grad_output.data
        else:
            grad_b = -1 * unbroadcast(b.data, grad_output.data)

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        if (a.data.shape != b.data.shape) and not checkBroadcast(a.data, b.data):
            raise Exception("Both args must have same size or should be broadcastable: {}, {}".format(a.shape, b.shape ))

        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor ( a.data * b.data, requires_grad=requires_grad,
                            is_leaf=not requires_grad )
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        grad_a = b.data * grad_output.data
        grad_b = a.data * grad_output.data

        if a.shape != (grad_a).shape:
            grad_a = unbroadcast(a.data, grad_a)

        if b.shape != (grad_b).shape:
            grad_b = unbroadcast(b.data, grad_b)

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        if (a.data.shape != b.data.shape) and not checkBroadcast ( a.data, b.data ):
            raise Exception ("Both args must have same size or should be broadcastable: {}, {}".format ( a.shape, b.shape ) )

        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data / b.data, requires_grad=requires_grad,
                            is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        grad_a = (np.ones(a.data.shape) * grad_output.data) / b.data
        grad_b = (-1 * a.data * grad_output.data) / np.square(b.data)

        if a.shape != (grad_a).shape:
            grad_a = unbroadcast(a.data, grad_a)

        if b.shape != (grad_b).shape:
            grad_b = unbroadcast(b.data, grad_b)

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors with appropriate dimensions
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        if a.data.shape[-1] != b.data.shape[0]:
            raise Exception ( "Dimension Mismatch for matmul: {}, {}".format(a.shape[-1], b.shape[0] ))

        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor ( a.data @ b.data, requires_grad=requires_grad,
                            is_leaf=not requires_grad )
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        grad_a = grad_output.data @ b.data.T
        grad_b = a.data.T @ grad_output.data

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for ReLU must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.maximum(a.data, 0), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(np.where(a.data > 0, 1.0, 0.0) * grad_output.data)

class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for ReLU must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.sum(a.data, axis=axis, keepdims=keepdims), requires_grad=requires_grad, is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        a_grad = np.broadcast_to(grad_output.data, a.shape)
        assert a_grad.shape == a.shape
        return tensor.Tensor(a_grad)

class Square(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for ReLU must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.square(a.data), requires_grad=requires_grad, is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        a_grad = 2 * a.data * grad_output.data
        assert a_grad.shape == a.shape
        return tensor.Tensor(a_grad)

class Power(Function):
    @staticmethod
    def forward(ctx, a, exp):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for ReLU must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        ctx.exp = exp
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.power(a.data, exp), requires_grad=requires_grad, is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        exp = ctx.exp
        a_grad = exp * np.power(a.data, exp-1) * grad_output.data
        assert a_grad.shape == a.shape
        return tensor.Tensor(a_grad)

class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for ReLU must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.sqrt(a.data), requires_grad=requires_grad, is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        a_grad = 0.5 * (grad_output.data / np.sqrt(a.data))
        assert a_grad.shape == a.shape
        return tensor.Tensor(a_grad)

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for ReLU must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.exp(a.data), requires_grad=requires_grad, is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        a_grad = np.exp(a.data) * grad_output.data
        assert a_grad.shape == a.shape
        return tensor.Tensor(a_grad)


class Dropout(Function):
    @staticmethod
    def forward(ctx, x, p=0.5, is_train=False):
        """Forward pass for dropout layer.

        Args:
            ctx (ContextManager): For saving variables between forward and backward passes.
            x (Tensor): Data tensor to perform dropout on
            p (float): The probability of dropping a neuron output.
                       (i.e. 0.2 -> 20% chance of dropping)
            is_train (bool, optional): If true, then the Dropout module that called this
                                       is in training mode (`<dropout_layer>.is_train == True`).

                                       Remember that Dropout operates differently during train
                                       and eval mode. During train it drops certain neuron outputs.
                                       During eval, it should NOT drop any outputs and return the input
                                       as is. This will also affect backprop correspondingly.
        """
        if not type ( x ).__name__ == 'Tensor':
            raise Exception ( "Only dropout for tensors is supported" )
        if is_train == True:
            requires_grad = x.requires_grad
            drop = tensor.Tensor(np.random.binomial(1, 1-p, x.data.shape), requires_grad=False)
            ctx.save_for_backward(x, drop)
            ctx.is_train = is_train
            ctx.p = p
            c = tensor.Tensor((x.data * drop.data)/(1-p), requires_grad=requires_grad, is_leaf=not requires_grad)
            return c
        else:
            ctx.save_for_backward(x)
            ctx.is_train = is_train
            return x

    @staticmethod
    def backward(ctx, grad_output):
        is_train = ctx.is_train
        if is_train:
            x, drop = ctx.saved_tensors
            p = ctx.p
            x_grad = (drop.data * grad_output.data)/(1-p)
            assert x_grad.shape == x.shape
            return tensor.Tensor (x_grad)
        else:
            x = ctx.saved_tensors[0]
            assert grad_output.shape == x.shape
            return grad_output


class Conv1d ( Function ):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
        """The forward/backward of a Conv1d Layer in the comp graph.

        Notes:
            - Make sure to implement the vectorized version of the pseudocode
            - See Lec 10 slides
            - No, you won't need to implement Conv2d for this homework.

        Args:
            x (Tensor): (batch_size, in_channel, input_size) input data
            weight (Tensor): (out_channel, in_channel, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution

        Returns:
            Tensor: (batch_size, out_channel, output_size) output data
        """
        # For your convenience: ints for each size
        batch_size, in_channels, input_size = x.shape
        out_channels, _, kernel_size = weight.shape

        ctx.save_for_backward ( x, weight, bias )
        ctx.stride = stride
        ctx.kernel_size = kernel_size
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels

        output_size = get_conv1d_output_size(input_size, kernel_size, stride)
        out = np.zeros(shape=(batch_size, out_channels, output_size))

        # Calculate the Conv1d output.
        # Remember that we're working with np.arrays; no new operations needed.
        for out_channel in range(out_channels):
            elem = 0
            for i in range ( 0, input_size - kernel_size + 1, stride ):
                out[:, out_channel, elem] = np.sum(np.sum(weight.data[out_channel] * x.data[:, :, i:i + kernel_size], axis = -1), axis = -1) + bias.data[out_channel]
                elem += 1


        # Put output into tensor with correct settings and return
        return tensor.Tensor(out, requires_grad=True, is_leaf=False)

    @staticmethod
    def backward(ctx, grad_output):
        # Finish Conv1d backward pass. It's surprisingly similar to the forward pass.
        stride = ctx.stride
        in_channels = ctx.in_channels
        kernel_size = ctx.kernel_size
        x, weight, bias = ctx.saved_tensors


        weight_flip = np.flip(np.flip(weight.data, axis = 2), axis = 1)

        grad_dilated = np.zeros ( shape=(grad_output.shape[0], grad_output.shape[1], grad_output.shape[2] + ((grad_output.shape[2] - 1) * (stride - 1))) )
        grad_dilated[:, :, ::stride] = grad_output.data

        # for gradients of input
        grad_dilated_x = grad_dilated.reshape ( grad_dilated.shape[0], grad_dilated.shape[1], 1, grad_dilated.shape[2])
        grad_dilated_x = np.pad(grad_dilated_x, ((0, 0), (0, 0), (in_channels - 1, in_channels - 1), (kernel_size - 1, kernel_size - 1)))
        x_grad = np.zeros(shape=x.shape)
        for j in range ( 0, grad_dilated_x.shape[2] - weight_flip.shape[1] + 1, 1 ): #in_channel
            for i in range ( 0, grad_dilated_x.shape[3] - weight_flip.shape[2] + 1, 1 ): #index of input element in row
                x_grad[:, j, i] = np.sum(np.sum(np.sum(weight_flip * grad_dilated_x[:, :, j:j + weight_flip.shape[1], i:i + weight_flip.shape[2]], axis=-1),axis=-1), axis=-1)

        # for gradients of weight
        grad_dil_w = grad_dilated
        w_grad = np.zeros ( shape=weight.shape)
        for c_in in range ( 0, x.shape[1], 1 ):  # number of in_channels
            for i in range ( 0, x.shape[2] - grad_dil_w.shape[2] + 1, 1 ):  # index of weight in a row
                if i >= w_grad.shape[2]:
                    break
                else:
                    for c_out in range ( weight.shape[0] ):  # number of out_channels
                        w_grad[c_out, c_in, i] = np.sum ( grad_dil_w[:, c_out] * x.data[:, c_in, i:i + grad_dil_w.shape[2]])

        # for gradients of bias
        b_grad = np.sum(np.sum(grad_output.data, axis=0), axis=1)

        return tensor.Tensor (x_grad), tensor.Tensor(w_grad), tensor.Tensor(b_grad)



class Conv2d ( Function ):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
        """
                Args:
                    x (Tensor): (batch_size, in_channel, input_size, input_size) input data
                    weight (Tensor): (out_channel, in_channel, kernel_size, kernel_size)
                    bias (Tensor): (out_channel,)
                    stride (int): Stride of the convolution

                Returns:
                    Tensor: (batch_size, out_channel, output_size, output_size) output data
                """
        batch_size, in_channels, input_size, _ = x.shape
        out_channels, _, kernel_size, _ = weight.shape

        ctx.save_for_backward ( x, weight, bias )
        ctx.stride = stride
        ctx.kernel_size = kernel_size
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels

        output_size = get_conv1d_output_size(input_size, kernel_size, stride)
        out = np.zeros(shape=(batch_size, out_channels, output_size, output_size))

        for out_channel in range(out_channels):
            elem_j = 0
            for j in range (0, input_size - kernel_size + 1, stride):
                elem_i = 0
                for i in range(0, input_size - kernel_size + 1, stride):
                    out[:, out_channel, elem_j, elem_i] = np.sum(np.sum(np.sum(weight.data[out_channel] * x.data[:, :, j:j + kernel_size, i:i + kernel_size], axis = -1), axis = -1), axis = -1) + bias.data[out_channel]
                    elem_i += 1
                elem_j += 1

        return tensor.Tensor(out, requires_grad=True, is_leaf=False)

    @staticmethod
    def backward(ctx, grad_output):
        stride = ctx.stride
        in_channels = ctx.in_channels
        out_channels = ctx.out_channels
        kernel_size = ctx.kernel_size
        x, weight, bias = ctx.saved_tensors

        weight_flip = np.flip(np.flip(weight.data, axis=3), axis=2)
        grad_dilated = np.zeros ( shape=(grad_output.shape[0], grad_output.shape[1],
                                   grad_output.shape[2] + ((grad_output.shape[2] - 1) * (stride - 1)),
                                    grad_output.shape[3] + ((grad_output.shape[3] - 1) * (stride - 1))))
        grad_dilated[:, :, ::stride, ::stride] = grad_output.data

        # for gradients of input
        grad_dilated_x = np.pad ( grad_dilated, ((0, 0), (0, 0), (kernel_size - 1, kernel_size - 1), (kernel_size - 1, kernel_size - 1)))
        x_grad = np.zeros ( shape=x.shape )
        for channel in range(in_channels):  # in_channel
            for j in range(x.shape[2]):
                for i in range(x.shape[3]):  # index of input element in row
                    if weight_flip.shape[2] != grad_dilated_x[:, :, j:j + weight_flip.shape[2], i:i + weight_flip.shape[3]].shape[2] or weight_flip.shape[3] != grad_dilated_x[:, :, j:j + weight_flip.shape[2], i:i + weight_flip.shape[3]].shape[3]:
                        break
                    else:
                        x_grad[:, channel, j, i] = np.sum(np.sum(np.sum(weight_flip[:, channel, :, :] * grad_dilated_x[:, :, j:j + weight_flip.shape[2], i:i + weight_flip.shape[3]], axis=-1), axis=-1), axis=-1)

        # for gradients of weight
        grad_dil_w = grad_dilated
        w_grad = np.zeros ( shape=weight.shape )
        for c_in in range (in_channels):
            for j in range(0, x.shape[2] - grad_dil_w.shape[2] + 1, 1):  # index of weight in a row
                for i in range ( 0, x.shape[3] - grad_dil_w.shape[3] + 1, 1 ):
                    if i >= w_grad.shape[3] or j >= w_grad.shape[2]:
                        break
                    else:
                        for c_out in range(out_channels):  # number of out_channels
                            w_grad[c_out, c_in, j, i] = np.sum(grad_dil_w[:, c_out] * x.data[:, c_in, j:j + grad_dil_w.shape[2], i:i + grad_dil_w.shape[3]])

        # for gradients of bias
        b_grad = np.sum(np.sum(np.sum(grad_output.data, axis=0), axis=1), axis=1)
        return tensor.Tensor (x_grad), tensor.Tensor(w_grad), tensor.Tensor(b_grad)

class MaxPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride=None):
        """
        Args:
            x (Tensor): (batch_size, in_channel, input_height, input_width)
            kernel_size (int): the size of the window to take a max over
            stride (int): the stride of the window. Default value is kernel_size.
        Returns:
            y (Tensor): (batch_size, out_channel, output_height, output_width)
        """
        batches, channels, input_size, _ = x.shape
        ctx.batches = batches
        ctx.channels = channels
        ctx.input_size = input_size
        ctx.kernel_size = kernel_size
        ctx.stride = stride

        output_size = get_conv1d_output_size(input_size, kernel_size, stride)
        out = np.zeros(shape=(batches, channels, output_size, output_size))
        for batch in range(batches):
            for channel in range(channels):
                elem_j = 0
                for j in range(0, input_size - kernel_size + 1, stride ):
                    elem_i = 0
                    for i in range(0, input_size - kernel_size + 1, stride):
                        out[batch, channel, elem_j, elem_i] = np.amax(
                            x.data[batch, channel, j:j + kernel_size, i:i + kernel_size] )
                        elem_i += 1
                    elem_j += 1
        out = tensor.Tensor(out, requires_grad=x.requires_grad, is_leaf=not x.requires_grad)
        ctx.save_for_backward (x, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            ctx (autograd_engine.ContextManager): for receiving objects you saved in this Function's forward
            grad_output (Tensor): (batch_size, out_channel, output_height, output_width)
                                  grad. of loss w.r.t. output of this function

        Returns:
            dx, None, None (tuple(Tensor, None, None)): Gradients of loss w.r.t. input
                                                        `None`s are to match forward's num input args
                                                        (This is just a suggestion; may depend on how
                                                         you've written `autograd_engine.py`)
        """
        x, out = ctx.saved_tensors
        batches = ctx.batches
        channels = ctx.channels
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        input_size = ctx.input_size

        x_grad = np.zeros(shape=x.shape)

        for batch in range ( batches ):
            for channel in range (channels):
                elem_j = 0
                for j in range ( 0, input_size - kernel_size + 1, stride ):
                    elem_i = 0
                    for i in range ( 0, input_size - kernel_size + 1, stride ):
                        locations = np.asarray ( np.where (
                            x.data[batch, channel, j:j + kernel_size, i:i + kernel_size] == out.data[
                                batch, channel, elem_j, elem_i] ) ).T
                        for ind in range(len(locations)):
                            x_grad[batch, channel, j + locations[ind][0], i + locations[ind][1]] += grad_output.data[
                                batch, channel, elem_j, elem_i]
                        elem_i += 1
                    elem_j += 1

        return tensor.Tensor(x_grad)



class AvgPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride=None):
        """
        Args:
            x (Tensor): (batch_size, in_channel, input_height, input_width)
            kernel_size (int): the size of the window to take a mean over
            stride (int): the stride of the window. Default value is kernel_size.
        Returns:
            y (Tensor): (batch_size, out_channel, output_height, output_width)
        """
        batches, channels, input_size, _ = x.shape
        ctx.batches = batches
        ctx.channels = channels
        ctx.input_size = input_size
        ctx.kernel_size = kernel_size
        ctx.stride = stride

        output_size = get_conv1d_output_size(input_size, kernel_size, stride)
        out = np.zeros(shape=(batches, channels, output_size, output_size))
        for batch in range(batches):
            for channel in range(channels):
                elem_j = 0
                for j in range(0, input_size - kernel_size + 1, stride ):
                    elem_i = 0
                    for i in range(0, input_size - kernel_size + 1, stride):
                        out[batch, channel, elem_j, elem_i] = np.mean(
                            x.data[batch, channel, j:j + kernel_size, i:i + kernel_size])
                        elem_i += 1
                    elem_j += 1
        out = tensor.Tensor(out, requires_grad=x.requires_grad, is_leaf=not x.requires_grad)
        ctx.save_for_backward (x, out)
        return out


    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            ctx (autograd_engine.ContextManager): for receiving objects you saved in this Function's forward
            grad_output (Tensor): (batch_size, out_channel, output_height, output_width)
                                  grad. of loss w.r.t. output of this function

        Returns:
            dx, None, None (tuple(Tensor, None, None)): Gradients of loss w.r.t. input
                                                        `None`s are to match forward's num input args
                                                        (This is just a suggestion; may depend on how
                                                         you've written `autograd_engine.py`)
        """
        x, out = ctx.saved_tensors
        batches = ctx.batches
        channels = ctx.channels
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        input_size = ctx.input_size

        x_grad = np.zeros ( shape=x.shape )

        for batch in range ( batches ):
            for channel in range ( channels ):
                elem_j = 0
                for j in range ( 0, input_size - kernel_size + 1, stride ):
                    elem_i = 0
                    for i in range ( 0, input_size - kernel_size + 1, stride ):
                        x_grad[batch, channel, j:j + kernel_size, i:i + kernel_size] += grad_output.data[batch, channel, elem_j, elem_i]/(np.square(kernel_size))
                        elem_i += 1
                    elem_j += 1

        return tensor.Tensor ( x_grad )



class Slice(Function):
    @staticmethod
    def forward(ctx,x,indices):
        '''
        Args:
            x (tensor): Tensor object that we need to slice
            indices (int,list,Slice): This is the key passed to the __getitem__ function of the Tensor object when it is sliced using [ ] notation.
        '''

        ctx.save_for_backward(x)
        ctx.indices = indices
        requires_grad = x.requires_grad
        if isinstance(indices, (int, slice, tuple)):
            c = x.data[indices]
        else:
            raise Exception("Invalid indices")
        return tensor.Tensor(c, requires_grad=requires_grad, is_leaf=not requires_grad)


    @staticmethod
    def backward(ctx,grad_output):
        indices = ctx.indices
        x = ctx.saved_tensors[0]
        grad = np.zeros(x.shape)
        grad[indices] = grad_output.data
        return tensor.Tensor(grad)


class Cat(Function):
    @staticmethod
    def forward(ctx, *args):
        '''
        Args:
            dim (int): The dimension along which we concatenate our tensors
            seq (list of tensors): list of tensors we wish to concatenate
        '''
        *seq, dim = args
        ctx.dim = dim
        np_seq = [t.data for t in seq]
        ctx.np_seq = np_seq
        length_list = np.cumsum([t.shape[dim] for t in np_seq])
        ctx.length_list = length_list

        requires_grad = any(t.requires_grad for t in seq)
        c = tensor.Tensor(np.concatenate(np_seq, axis = dim), requires_grad=requires_grad, is_leaf=not requires_grad)
        return c


    @staticmethod
    def backward(ctx,grad_output):
        dim = ctx.dim
        length_list = ctx.length_list
        np_grad_list = np.split(grad_output.data, length_list[:-1], axis=dim)
        tensor_grad_list = [tensor.Tensor(t) for t in np_grad_list]
        return (*tensor_grad_list, None)

def get_conv1d_output_size(input_size, kernel_size, stride):
    """Gets the size of a Conv1d output.

    Notes:
        - This formula should NOT add to the comp graph.
        - Yes, Conv2d would use a different formula,
        - But no, you don't need to account for Conv2d here.

        - If you want, you can modify and use this function in HW2P2.
            - You could add in Conv1d/Conv2d handling, account for padding, dilation, etc.
            - In that case refer to the torch docs for the full formulas.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution

    Returns:
        int: size of the output as an int (not a Tensor or np.array)
    """
    return ((input_size - kernel_size)//stride) + 1


def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """
    batch_size, num_classes = predicted.shape
    one_hots = to_one_hot(target, num_classes)
    max = tensor.Tensor(np.max(predicted.data), requires_grad=False)
    logsoftmax = predicted - (predicted - max).exp().sum(axis = 1, keepdims = True).log() - max
    return ((tensor.Tensor(np.array([-1.]), requires_grad=False) * (one_hots * logsoftmax).sum(axis = 1, keepdims=True)).sum(axis = 0) / tensor.Tensor(np.array([batch_size]), requires_grad=False)).reshape()

    # Tip: You can implement XELoss all here, without creating a new subclass of Function.
    #      However, if you'd prefer to implement a Function subclass you're free to.
    #      Just be sure that nn.loss.CrossEntropyLoss calls it properly.

    # Tip 2: Remember to divide the loss by batch_size; this is equivalent
    #        to reduction='mean' in PyTorch's nn.CrossEntropyLoss

def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]
     
    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad = True)


class Sigmoid ( Function ):
    @staticmethod
    def forward(ctx, a):
        b_data = np.divide ( 1.0, np.add ( 1.0, np.exp ( -a.data ) ) )
        ctx.out = b_data[:]
        b = tensor.Tensor ( b_data, requires_grad=a.requires_grad )
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        b = ctx.out
        grad = grad_output.data * b * (1 - b)
        return tensor.Tensor(grad)


class Tanh ( Function ):
    @staticmethod
    def forward(ctx, a):
        b = tensor.Tensor ( np.tanh ( a.data ), requires_grad=a.requires_grad )
        ctx.out = b.data[:]
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.out
        grad = grad_output.data * (1 - out ** 2)
        return tensor.Tensor ( grad )



