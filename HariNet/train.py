'''
train a nn
'''

from HariNet.tensor import Tensor
from HariNet.nn import NeuralNet
from HariNet.loss import Loss,MSE
from HariNet.optimizers import Optimizer,SGD
from HariNet.data import DataIter, BatchIterator

def train(network: NeuralNet,
          inputs: Tensor,
          targets:Tensor,
          num_epochs:int = 1000,
          iterator:DataIter = BatchIterator(),
          loss:Loss = MSE(),
          optim: Optimizer = SGD()):
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs,targets):
            predicted = network.forward(batch.inputs)
            epoch_loss += loss.loss(predicted,batch.targets)
            grad = loss.grad(predicted,batch.targets)
            network.backprop(grad)
            optim.step(network)
        print(epoch,epoch_loss)