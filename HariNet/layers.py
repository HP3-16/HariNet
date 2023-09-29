from HariNet.tensor import Tensor
import numpy as np
from typing import Dict


class Layer:
    def __init__(self) -> None:
        self.params: Dict = {}

    def forward(self,inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backprop(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_size: int,output_size: int) -> None:
        self.params["w"] = np.random.randn(input_size,output_size)
        self.params["b"] = np.random.rand(output_size)



