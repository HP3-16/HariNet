from typing import Sequence
from HariNet.tensor import Tensor
from HariNet.layers import Layer

class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers
    
    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        
        return inputs
    
    def backprop(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backprop(grad)
        return grad
    
    def paramgrad(self):
        for layer in self.layers:
            for name,param in layer.params.items():
                grad = layer.grads[name]
                yield param,grad