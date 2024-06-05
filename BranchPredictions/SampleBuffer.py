import numpy as np
import torch
import time

DEBUG_TIMING = {
    'SampleBufferCreate': [],
    'SampleBufferCalcSize': [],
    'SampleBufferExtract': [],
}


class SamplingBuffer:

    def __init__(self, device, numShuffleSplits=128, bufferSize=10000000):
        self.bufferSize = bufferSize
        self.device = device

        self.randBuffer = None
        self.randBufferIndex = 0
        self.shapeRandBuffers = {}
        self.numShuffleSplits = numShuffleSplits

    def createShapeBufferData(self, shape, epsilon=1e-14):
        dimShuffleSplits = []
        for i in range(len(shape) - 1):
            dimSize = shape[i + 1]
            dimShuffleSplits.append(np.random.randint(0, dimSize - 1, self.numShuffleSplits))
        noise = -torch.log(torch.rand(shape) + epsilon).to(self.device)
        return [noise, dimShuffleSplits, 0]

    def getShapeRandBuffers(self, shape):
        if (shape not in self.shapeRandBuffers):
            self.shapeRandBuffers[shape] = self.createShapeBufferData(shape)

        noise, dimShuffleSplits, sampleCounter = self.shapeRandBuffers[shape]
        for i in range(len(shape) - 1):
            split = dimShuffleSplits[i][sampleCounter]
            firstSlice = [slice(None)] * (i + 1) + [slice(split, None)]
            secondSlice = [slice(None)] * (i + 1) + [slice(0, split)]
            noise = torch.cat((noise[firstSlice], noise[secondSlice]), dim=i + 1)
        self.shapeRandBuffers[shape][-1] = (sampleCounter + 1) % self.numShuffleSplits

        return noise

    def rand(self, shape):
        t2 = time.time()
        randNoise = self.getShapeRandBuffers(shape)
        t3 = time.time()

        # DEBUG_TIMING['SampleBufferCreate'].append(t1 - t0)
        # DEBUG_TIMING['SampleBufferCalcSize'].append(t2 - t1)
        DEBUG_TIMING['SampleBufferExtract'].append(t3 - t2)
        return randNoise


if __name__ == '__main__':
    # shape = (8, 63, 512)
    storageDict = {}
    shape = (1, 4, 4)

    if (shape not in storageDict):
        numSamples = torch.prod(torch.tensor(shape)).item()
        indexTensor = torch.randperm(numSamples).view(shape)

        storageDict[shape] = indexTensor

    target = storageDict[shape]
    print(target)
    for i in range(len(shape) - 1):
        split = 2
        firstSlice = [slice(None)] * (i + 1) + [slice(split, None)]
        secondSlice = [slice(None)] * (i + 1) + [slice(0, split)]
        target = torch.cat((target[firstSlice], target[secondSlice]), dim=i + 1)

    print(target)
    storageDict[shape] = target

    print(storageDict[shape])
