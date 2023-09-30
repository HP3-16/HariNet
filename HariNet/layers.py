from HariNet.tensor import Tensor
import numpy as np
from typing import Dict
from collections.abc import Callable


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str,Tensor] = {}

    def forward(self,inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backprop(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_size:int, output_size:int) -> None:
        super().__init__()
        self.params["w"] = np.random.randn(input_size,output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"] # @ is the __matmul__
    
    def backprop(self, grad: Tensor) -> Tensor:
        '''
        if y = f(x) and x = a @ b + c
        dy/da = f'(x) @ b.T
        dy/db = a.T @ f'(x)
        dy/dc = f'(x)

        x = i*w + b
        y = f(x)
        dy/di
        dy/dw
        dy/db   
        '''
        self.grads["b"] = np.sum(grad,axis=0) #grad w.r.t  b
        self.grads["w"] = self.inputs.T @ grad # grad w.r.t w
        return grad @ self.params["w"].T # grad w.r.t inputs to the layer

Function = Callable[[Tensor],Tensor]
class Activation(Layer):
    '''
    applies a function to input
    '''

    def __init__(self, f: Function, fdash: Function) -> None:
        super().__init__()
        self.f = f
        self.fdash = fdash

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)
    
    def backprop(self, grad: Tensor) -> Tensor:
        return self.fdash(self.inputs) * grad
    
def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)
    
    
def tanhdash(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y**2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh,tanhdash)
        


