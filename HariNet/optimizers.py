'''
adjusting parameters of nn, based on gradients computed
'''
from HariNet.nn import NeuralNet
class Optimizer:
    def step(self, network: NeuralNet) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr = 0.01) -> None:
        self.lr = lr
    
    def step(self,network: NeuralNet) -> None:
        for param,grad in network.paramgrad():
            param -= self.lr * grad
