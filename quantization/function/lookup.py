import torch


class Lookup(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        """
        在前向传播中，我们接收一个Tensor包含输入，返回一个Tensor包含输出。ctx是一个上下文对象，
        可以用来存放反向计算的信息，可以用ctx.save_for_backward方法缓存任意在反向传播中用到的对象
        """
        ctx.save_for_backward(input)  # 这里input是x.mm(w1)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播中，我们接收一个Tensor包含loss对于输出的梯度，需要计算loss对于输入的梯度
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

