'''
batch iteration
'''
import numpy as np
from typing import NamedTuple
from HariNet.tensor import Tensor

Batch = NamedTuple("Batch",[("inputs",Tensor),("targets",Tensor)])

class DataIter:
    def __call__(self, input, targets):
        raise NotImplementedError
    
class BatchIterator(DataIter):
    def __init__(self, batch_size = 16, shuffle = True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs, targets):
        starts = np.arange(0,len(inputs),self.batch_size)

        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs,batch_targets)


