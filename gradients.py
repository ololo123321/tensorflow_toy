import numpy as np
from base import Op, Variable, Placeholder


def traverse(op):
    res = []

    def f(node):
        if isinstance(node, Op):
            for operand in node.operands:
                f(operand)
        res.append(node)
    f(op)
    return res


def forward(op, feed_dict):
    nodes = traverse(op)
    for node in nodes:
        if isinstance(node, Variable):
            continue
        elif isinstance(node, Placeholder):
            node.value = feed_dict[node]
        else:
            node.forward()
    return op.value


def gradients(op, feed_dict):
    grads = {}
    nodes = traverse(op)

    # forward pass:
    for node in nodes:
        if isinstance(node, Variable):
            continue
        elif isinstance(node, Placeholder):
            node.value = feed_dict[node]
        else:
            node.forward()

    # backward pass:
    op.grad = np.ones_like(op.value)
    for node in reversed(nodes):
        if isinstance(node, Op):
            node.backward()
        grads[node] = node.grad

    return grads


class GradientDescentOptimizer:
    def __init__(self, learning_rate=1e-2):
        self.learning_rate = learning_rate

    def minimize(self, op, feed_dict):
        grads = gradients(op, feed_dict)
        for node, grad in grads.items():
            if not isinstance(node, Placeholder):
                node.value -= self.learning_rate * grad
