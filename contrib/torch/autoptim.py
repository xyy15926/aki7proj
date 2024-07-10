#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: autograd.py
#   Author: xyy15926
#   Created: 2024-06-25 18:27:06
#   Updated: 2024-06-26 20:11:10
#   Description:
#
#  Torch provides two main features:
#  1. An n-dimensional Tesnor, similar to numpy.ndarray but can run on GPUs.
#  2. Automatics differentiation for building and traing NN.
#  Reference:
#    - <https://pytorch.org/tutorials/beginner/pytorch_with_examples.html>
#    - <https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html>
#    - <https://pytorch.org/docs/stable/notes/autograd.html>
# ---------------------------------------------------------

# %%
import logging
import numpy as np

import torch
from torch import nn, optim, autograd

DEVICE = ("cuda"
          if torch.cuda.is_available()
          else "mps"
          if torch.backends.mps.is_available()
          else "cpu")
CLASSES = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

logging.basicConfig(
    # format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    format="%(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
# Reference: <https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#warm-up-numpy>
def np_backward_propagation():
    """Backward propagation implemented with NumPy.

    1. Backprop to compute gradient of parameters with respect to loss.
    2. Update parameters with gradient.
    """
    x = np.linspace(-np.pi, np.pi, 2000)
    y = np.sin(x)
    a, b, c, d = np.random.randn(4)
    lr = 1e-6
    epochs = 2000

    for t in range(epochs):
        y_pred = a + b * x + c * x ** 2 + d * x ** 3
        loss = np.square(y_pred - y).sum()

        if t % 100 == 99:
            logger.info(f"Loss: {loss:>7f} [{t:>5d} / {epochs:>5d}]")

        # Backprop to compute gradient of parameters with respect to loss.
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()

        # Update parameters.
        a -= lr * grad_a
        b -= lr * grad_b
        c -= lr * grad_c
        d -= lr * grad_d

    logger.info(f"Backprop Result: y = {a} + {b}x + {c}x^2 + {d}x^3")
    logger.info(f"Final Loss: {loss}")


# %%
def torch_backward_propagation():
    """Take the advantage of the PyTorch's autograd to compute gradient.
    """
    x = torch.linspace(-np.pi, np.pi, 2000)
    y = torch.sin(x)
    lr = 1e-6
    epochs = 2000

    # Set parameters with `requires_grad=True`.
    a = torch.randn((), dtype=torch.float, requires_grad=True)
    b = torch.randn((), dtype=torch.float, requires_grad=True)
    c = torch.randn((), dtype=torch.float, requires_grad=True)
    d = torch.randn((), dtype=torch.float, requires_grad=True)

    for t in range(epochs):
        y_pred = a + b * x + c * x ** 2 + d * x ** 3
        loss = (y_pred - y).pow(2).sum()

        if t % 100 == 99:
            logger.info(f"Loss: {loss:>7f} [{t:>5d} / {epochs:>5d}]")

        # Use autograd to compute backward pass.
        # Gradient of loss of all Tensors with `requires_grad = True` will be
        # computed and held in `.grad`.
        loss.backward()

        # Update parameters.
        with torch.no_grad():
            a -= lr * a.grad
            b -= lr * b.grad
            c -= lr * c.grad
            d -= lr * d.grad

            # Zero `grad` manually or it will be accumulated.
            a.grad.zero_()
            b.grad.zero_()
            c.grad.zero_()
            d.grad.zero_()

    logger.info(f"Backprop Result: y = {a} + {b}x + {c}x^2 + {d}x^3")
    logger.info(f"Final Loss: {loss}")


# %%
# Only `forward`, `backward` method are necessary to implemented to define
# customed autograd funtion.
#
# Static method `backward` of autograd function will be called when attaching
# `tensor.backward()` is called. And it could be assumed that 1 will be passed
# as the second parameter if `gradient` in `tensor.backward()` is not provided.
class LegendrePolynominal3(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """Forward pass.

        Params:
        --------------------------
        ctx: Context to stash information to backward computation.
        input: Tensor containing the input.

        Return:
        --------------------------
        Tensor containing the output.
        """
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass.

        Params:
        --------------------------
        ctx: Context to stash information to backward computation.
        grad_output: Tensor containing the gradient of the loss with respect
          to the output.

        Return:
        --------------------------
        Tensor containing the gradient of the loss with respect to the input.
        """
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)


def torch_backprop_with_customed_func():
    """Fit with cumtomed autograd function.
    """
    x = torch.linspace(-np.pi, np.pi, 2000)
    y = torch.sin(x)
    lr = 1e-6
    epochs = 2000

    # Set parameters with `requires_grad=True`.
    # Here initial values are set carefully.
    a = torch.full((), 0.0, dtype=torch.float, requires_grad=True)
    b = torch.full((), -1.0, dtype=torch.float, requires_grad=True)
    c = torch.full((), 0.0, dtype=torch.float, requires_grad=True)
    d = torch.full((), 0.0, dtype=torch.float, requires_grad=True)

    for t in range(epochs):
        P3 = LegendrePolynominal3.apply
        y_pred = a + b * x + P3(c + d * x)
        loss = (y_pred - y).pow(2).sum()

        if t % 100 == 99:
            logger.info(f"Loss: {loss:>7f} [{t:>5d} / {epochs:>5d}]")

        # Use autograd to compute backward pass.
        # Gradient of loss of all Tensors with `requires_grad = True` will be
        # computed and held in `.grad`.
        loss.backward()

        # Update parameters.
        with torch.no_grad():
            a -= lr * a.grad
            b -= lr * b.grad
            c -= lr * c.grad
            d -= lr * d.grad

            a.grad.zero_()
            b.grad.zero_()
            c.grad.zero_()
            d.grad.zero_()

    logger.info(f"Backprop Result: y = {a} + {b}x + P3({c} + {d}x)")
    logger.info(f"Final Loss: {loss}")


# %%
# Reference: <https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html>
def autograd_scalar():
    x = torch.ones(5)
    y = torch.zeros(3)
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    bb = torch.randn(3)
    z = torch.matmul(x, w) + b + bb
    loss = nn.functional.binary_cross_entropy_with_logits(z, y)

    assert b.requires_grad
    assert not bb.requires_grad

    z_det = z.detach()
    assert not z_det.requires_grad

    loss.backward()
    assert b.grad is not None


def autograd_jacobian_product():
    inp = torch.eye(4, 5, requires_grad=True)
    out = (inp + 1).pow(2).t()
    # Grad can be implicitly created only for scalar outputs.
    out.backward(torch.ones_like(out), retain_graph=True)
    logger.info(f"First call: \n {inp.grad}")

    inp_sum = torch.eye(4, 5, requires_grad=True)
    out_sum = (inp_sum + 1).pow(2).t().sum()
    out_sum.backward(retain_graph=True)
    logger.info(f"First sum call: \n {inp_sum.grad}")
    assert torch.all(inp.grad == inp_sum.grad)

    out.backward(torch.ones_like(out), retain_graph=True)
    logger.info(f"call: \n {inp.grad}")
    assert torch.all(inp_sum.grad * 2 == inp.grad)

    # Gradient will be accumulated during backward propagation unless zeroed out
    # explicitly.
    inp.grad.zero_()
    out.backward(torch.ones_like(out), retain_graph=True)
    assert torch.all(inp.grad == inp_sum.grad)


# %%
def torch_nn_modules():
    """Fit with nn.Sequential modules.
    """
    x = torch.linspace(-np.pi, np.pi, 2000)
    y = torch.sin(x)
    p = torch.tensor([1, 2, 3])
    # Pow before input for convinience.
    xx = x.unsqueeze(-1).pow(p)

    lr = 1e-6
    epochs = 2000

    model = nn.Sequential(
        nn.Linear(3, 1),
        nn.Flatten(0, 1),
    )
    loss_fn = nn.MSELoss(reduction="sum")

    for t in range(epochs):
        y_pred = model(xx)
        loss = loss_fn(y_pred, y)

        if t % 100 == 99:
            logger.info(f"Loss: {loss:>7f} [{t:>5d} / {epochs:>5d}]")

        # Reset gradient of model parameters to prevent double counting.
        model.zero_grad()
        # Backpropagate the prediction loss.
        loss.backward()
        # Adjust parameters by the gradients collected in the backward pass.
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad

    ll = model[0]
    logger.info(f"Model Result: y = {ll.bias.item()} + {ll.weight[:, 0].item()}x +"
                f"{ll.weight[:, 1].item()}x^2 + {ll.weight[:, 2].item()}x^3")
    logger.info(f"Final Loss: {loss}")


# %%
def torch_optim():
    """Take the advantage of optimizer to manage the parameters' autograd.

    1. Call `optimizer.zero_grad()` to reset gradient of model parameters to
      prevent double counting.
    2. Call `loss.backward()` to backpropagate the prediction loss.
    3. Call `optimizer.step()` to adjust parameters by the gradients
      collected in the backward pass.
    """
    x = torch.linspace(-np.pi, np.pi, 2000)
    y = torch.sin(x)
    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)

    epochs = 2000

    model = nn.Sequential(
        nn.Linear(3, 1),
        nn.Flatten(0, 1),
    )
    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    for t in range(epochs):
        y_pred = model(xx)
        loss = loss_fn(y_pred, y)

        if t % 100 == 99:
            logger.info(f"Loss: {loss:>7f} [{t:>5d} / {epochs:>5d}]")

        # Reset gradient of model parameters to prevent double counting.
        optimizer.zero_grad()
        # Backpropagate the prediction loss.
        loss.backward()
        # Adjust parameters by the gradients collected in the backward pass.
        optimizer.step()

    ll = model[0]
    logger.info(f"Model Result: y = {ll.bias.item()} + {ll.weight[:, 0].item()}x "
                f"+ {ll.weight[:, 1].item()}x^2 + {ll.weight[:, 2].item()}x^3")
    logger.info(f"Final Loss: {loss}")


# %%
# For `optim.Optimizer`
#
# Attrs:
# -----------------------------
# param_groups: List[Dict]
#   Parameters could be partitioned into different groups so to attached to
#   different optimizer settings. And this stores the Dict of settings for
#   each parameter group. The keys in Dict are the parameters for `__init__`:
#   - params: Tensors to be optimized.
#   - lr: Learning rate for the corresponsible Tensors.
#
# Params for `__init__`:
# -----------------------------
# params: Iterable[Tensor | Dict[str, Tensor]]
#   Iterable[Tensor]: Tensors to be optimized.
#   Iterable[Dict]: Groups of Tensors to be optimized and theirs optimizer
#     settings. The keys in Dict must be the parameters for `__init__`:
#     - params: Tensors to be optimized.
#     - lr: Learning rate for the corresponsible Tensors.
# defaults: Dict[str, any]
#   Defaults values of optimization options, used when a parameter group
#   doesn't specify them.
#
class MySGD(optim.Optimizer):
    # Init the internal state.
    def __init__(self, params, lr=0.01):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    # Perform single optimization step to update tracked parameters.
    @torch.no_grad
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            # Traverse through all parameters to update with gradient.
            for p in group["params"]:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)


def torch_customed_optim():
    """Take the advantage of optimizer to manage the parameters' autograd.

    1. Call `optimizer.zero_grad()` to reset gradient of model parameters to
      prevent double counting.
    2. Call `loss.backward()` to backpropagate the prediction loss.
    3. Call `optimizer.step()` to adjust parameters by the gradients
      collected in the backward pass.
    """
    x = torch.linspace(-np.pi, np.pi, 2000)
    y = torch.sin(x)
    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)

    epochs = 2000

    model = nn.Sequential(
        nn.Linear(3, 1),
        nn.Flatten(0, 1),
    )
    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = MySGD(model.parameters(), lr=1e-6)

    for t in range(epochs):
        y_pred = model(xx)
        loss = loss_fn(y_pred, y)

        if t % 100 == 99:
            logger.info(f"Loss: {loss:>7f} [{t:>5d} / {epochs:>5d}]")

        # Reset gradient of model parameters to prevent double counting.
        optimizer.zero_grad()
        # Backpropagate the prediction loss.
        loss.backward()
        # Adjust parameters by the gradients collected in the backward pass.
        optimizer.step()

    ll = model[0]
    logger.info(f"Model Result: y = {ll.bias.item()} + {ll.weight[:, 0].item()}x "
                f"+ {ll.weight[:, 1].item()}x^2 + {ll.weight[:, 2].item()}x^3")
    logger.info(f"Final Loss: {loss}")


# %%
# Only `__init__` and `forward` are necessary to define customed Module.
# 1. `Parameter`s defined as the attributes of module will be tracked for
#   autograd with `.parameter()`. So are the `Parameter`s in predefined
#   modules serving as the attributes.
# 2. The method `forward` indicates the process from input to output.
class Polynomial3(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(()))
        self.b = nn.Parameter(torch.randn(()))
        self.c = nn.Parameter(torch.randn(()))
        self.d = nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        return (f"y = {self.a.item()} + {self.b.item()}x"
                f"+ {self.c.item()}x^2 + {self.d.item()}x^3")


def torch_nn_customed_module():
    """Fit with customed nn.Module.
    """
    x = torch.linspace(-np.pi, np.pi, 2000)
    y = torch.sin(x)

    epochs = 2000

    model = Polynomial3()
    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = optim.SGD(model.parameters(), lr=1e-6)

    for t in range(epochs):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        if t % 100 == 99:
            logger.info(f"Loss: {loss:>7f} [{t:>5d} / {epochs:>5d}]")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info(f"Result: {model.string()}")
    logger.info(f"Final Loss: {loss}")


# %%
# Dynamic graphs and weight sharing are allowed in PyTorch.
class DynPolynomial3(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(()))
        self.b = nn.Parameter(torch.randn(()))
        self.c = nn.Parameter(torch.randn(()))
        self.d = nn.Parameter(torch.randn(()))
        self.e = nn.Parameter(torch.randn(()))

    def forward(self, x):
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        # Python control-flow could be used to build dynamic graphs.
        # And parameters are reused.
        for exp in range(4, np.random.randint(4, 6)):
            y += self.e * x ** exp
        return y

    def string(self):
        return (f"y = {self.a.item()} + {self.b.item()}x "
                f"+ {self.c.item()}x^2 + {self.d.item()}x^3 "
                f"+ {self.e.item()}x^4? + {self.e.item()}x^5?")


def torch_nn_dyn_module():
    x = torch.linspace(-np.pi, np.pi, 2000)
    y = torch.sin(x)

    epochs = 4000

    model = DynPolynomial3()
    loss_fn = nn.MSELoss(reduction="sum")
    # Inproper hyper-params may lead to `nan` or `inf`.
    optimizer = optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)

    for t in range(epochs):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        if t % 100 == 99:
            logger.info(f"Loss: {loss:>7f} [{t:>5d} / {epochs:>5d}]")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info(f"Result: {model.string()}")
    logger.info(f"Final Loss: {loss}")


# %%
if __name__ == "__main__":
    np_backward_propagation()
    torch_backward_propagation()
    torch_backprop_with_customed_func()
    autograd_scalar()
    autograd_jacobian_product()
    torch_optim()
    torch_customed_optim()
    torch_nn_modules()
    torch_nn_customed_module()
    torch_nn_dyn_module()
