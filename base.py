class Placeholder:
    def __init__(self, name=None):
        self.value = None
        self.grad = None
        self.name = name


class Variable:
    def __init__(self, initial_value=None, name=None):
        self.value = initial_value
        self.grad = None
        self.name = name


class Op:
    def __init__(self, name=None):
        self.value = None  # результат вычисления Op
        self.grad = None  # производная по результату Op
        self.name = name

    def forward(self):
        pass

    def backward(self):
        pass
