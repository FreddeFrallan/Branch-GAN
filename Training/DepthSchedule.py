import copy

class DepthSchedule:

    def __init__(self, startDepth):
        self.currentDepth = startDepth

    def updateSchedule(self, epoch, batch, totalUpdates):
        raise NotImplementedError("DepthSchedule is an abstract class. Please use a concrete implementation.")

    def setSampleConfDepth(self, sampleConf):
        # Copy the sample configuration
        sampleConf = copy.deepcopy(sampleConf)

        print("Setting depth to: ", self.currentDepth)

        # Set the depth
        sampleConf.rules = sampleConf.rules[:self.currentDepth]
        if(self.currentDepth > 1):
            sampleConf.multiStepSampling = True

        return sampleConf


class LinearEpochDepthSchedule(DepthSchedule):
    def __init__(self, startDepth, incrementSize, epochsPerIncrement, maxDepth=None):
        super().__init__(startDepth)
        self.startDepth = startDepth
        self.incrementSize = incrementSize
        self.epochsPerIncrement = epochsPerIncrement
        self.maxDepth = maxDepth

    def updateSchedule(self, epoch, batch, totalUpdates):
        self.currentDepth = int(self.startDepth + (epoch // self.epochsPerIncrement) * self.incrementSize)
        if (self.maxDepth is not None):
            self.currentDepth = min(self.currentDepth, self.maxDepth)


class LinearUpdatesDepthSchedule(DepthSchedule):
    def __init__(self, startDepth, incrementSize, updatesPerIncrement, maxDepth):
        super().__init__(startDepth)
        self.startDepth = startDepth
        self.incrementSize = incrementSize
        self.updatesPerIncrement = updatesPerIncrement
        self.maxDepth = maxDepth

    def updateSchedule(self, epoch, batch, totalUpdates):
        self.currentDepth = int(self.startDepth + (totalUpdates // self.updatesPerIncrement) * self.incrementSize)
        if (self.maxDepth is not None):
            self.currentDepth = min(self.currentDepth, self.maxDepth)
