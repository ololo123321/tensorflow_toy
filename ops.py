import numpy as np
from base import Op


class add(Op):
    def __init__(self, x, y, name=None):
        super().__init__(name)
        self.x = x
        self.y = y
        self.operands = [x, y]

    def forward(self):
        self.value = self.x.value + self.y.value

    def backward(self):
        xv, yv = self.x.value, self.y.value
        self.x.grad = self.grad
        while np.ndim(self.x.grad) > len(xv.shape):
            self.x.grad = np.sum(self.x.grad, axis=0)
        for axis, size in enumerate(xv.shape):
            if size == 1:
                self.x.grad = np.sum(self.x.grad, axis=axis, keepdims=True)

        self.y.grad = self.grad
        while np.ndim(self.y.grad) > len(yv.shape):
            self.y.grad = np.sum(self.y.grad, axis=0)
        for axis, size in enumerate(yv.shape):
            if size == 1:
                self.y.grad = np.sum(self.y.grad, axis=axis, keepdims=True)


class power(Op):
    def __init__(self, x, y, name=None):
        super().__init__(name)
        self.x = x
        self.y = y
        self.operands = [x, y]

    def forward(self):
        self.value = self.x.value ** self.y.value

    def backward(self):
        xv, yv = self.x.value, self.y.value
        self.x.grad = yv * xv ** (yv - 1) * self.grad
        self.y.grad = xv ** yv * np.log(xv) * self.grad


class exp(Op):
    def __init__(self, x, name=None):
        super().__init__(name)
        self.x = x
        self.operands = [x]

    def forward(self):
        self.value = np.exp(self.x.value)

    def backward(self):
        self.x.grad = self.value * self.grad


class log(Op):
    def __init__(self, x, name=None):
        super().__init__(name)
        self.x = x
        self.operands = [x]

    def forward(self):
        self.value = np.log(self.x.value)

    def backward(self):
        self.x.grad = 1 / self.x.value * self.grad


class neg(Op):
    def __init__(self, x, name=None):
        super().__init__(name)
        self.x = x
        self.operands = [x]

    def forward(self):
        self.value = -self.x.value

    def backward(self):
        self.x.grad = -self.grad


class matmul(Op):
    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.a = a
        self.b = b
        self.operands = [a, b]

    def forward(self):
        self.value = self.a.value @ self.b.value

    def backward(self):
        self.a.grad = self.grad @ self.b.value.T
        self.b.grad = self.a.value.T @ self.grad


class multiply(Op):
    def __init__(self, x, y, name=None):
        super().__init__(name)
        self.x = x
        self.y = y
        self.operands = [x, y]

    def forward(self):
        self.value = self.x.value * self.y.value

    def backward(self):
        self.x.grad = self.grad * self.y.value
        self.y.grad = self.grad * self.x.value


class reduce_sum(Op):
    def __init__(self, x, axis=None, name=None):
        super().__init__(name)
        self.x = x
        self.axis = axis
        self.operands = [x]

    def forward(self):
        self.value = np.sum(self.x.value, axis=self.axis)

    def backward(self):
        input_shape = self.x.value.shape
        output_shape = np.array(input_shape)
        output_shape[self.axis] = 1
        tile_scaling = input_shape // output_shape
        self.grad = np.reshape(self.grad, output_shape)
        self.x.grad = np.tile(self.grad, tile_scaling)


class sigmoid(Op):
    def __init__(self, x, name=None):
        super().__init__(name)
        self.x = x
        self.operands = [x]

    def forward(self):
        self.value = 1 / (1 + np.exp(-self.x.value))

    def backward(self):
        self.x.grad = self.value * (1 - self.value) * self.grad


class softmax(Op):
    def __init__(self, x, name=None):
        super().__init__(name)
        self.x = x
        self.operands = [x]

    def forward(self):
        scores_exp = np.exp(self.x.value - np.max(self.x.value, axis=1, keepdims=True))
        self.value = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)

    def backward(self):
        self.x.grad = (self.grad - np.sum(self.grad * self.value, axis=1).reshape(-1, 1)) * self.value


class relu(Op):
    def __init__(self, x, name=None):
        super().__init__(name)
        self.x = x
        self.operands = [x]

    def forward(self):
        self.value = np.maximum(0, self.x.value)

    def backward(self):
        self.x.grad = (self.x.value >= 0).astype(float) * self.grad


class batchnorm(Op):
    def __init__(self, x, gamma, beta, mode='train', eps=1e-5, momentum=0.9, name=None):
        super().__init__(name)
        self.x = x
        self.gamma = gamma
        self.beta = beta
        self.operands = [x, gamma, beta]

        self.mode = mode
        self.eps = eps
        self.momentum = momentum

        self.running_mean = 0
        self.running_var = 0

        self.mu = None
        self.inv_std = None
        self.x_centered = None
        self.x_normed = None

    def forward(self):
        x, gamma, beta = self.x.value, self.gamma.value, self.beta.value
        if self.mode == 'train':
            self.mu = x.mean(axis=0)
            var = x.var(axis=0)
            self.inv_std = (var + self.eps) ** -0.5
            self.x_centered = x - self.mu
            self.x_normed = self.x_centered * self.inv_std
            self.value = self.x_hat * gamma + beta

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        elif self.mode == 'test':
            self.value = (x - self.running_mean) * (self.running_var + self.eps) ** -0.5
            self.value = self.value * gamma + beta
        else:
            raise ValueError('invalid mode')

    def backward(self):
        self.x.grad = self.inv_std * self.gamma * (
            self.grad - np.mean(self.grad, axis=0) -
            self.inv_std * self.x_normed * np.mean(self.grad * self.x_centered, axis=0)
        )
        self.gamma.grad = np.sum(self.x_normed * self.grad, axis=0)
        self.beta.grad = np.sum(self.grad, axis=0)
