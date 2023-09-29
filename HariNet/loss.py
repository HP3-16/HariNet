import numpy as np
from HariNet.tensor import Tensor

class Loss:
    def loss(self, yhat: Tensor, ytrue: Tensor) -> float:
        raise NotImplementedError
    
    def grad(self, yhat: Tensor, ytrue: Tensor) -> Tensor:
        raise NotImplementedError

class MSE(Loss):
    def loss(self, yhat: Tensor, ytrue: Tensor) -> float:
        return np.mean(np.sum((yhat-ytrue)**2))
    
    def grad(self, yhat: Tensor, ytrue: Tensor) -> Tensor:
        return 2 * (yhat - ytrue)
        