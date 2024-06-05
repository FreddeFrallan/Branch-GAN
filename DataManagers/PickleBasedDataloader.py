import numpy as np
import pickle
import torch
import tqdm


class IDsDataset(torch.utils.data.Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        ids = self.ids[item]
        return torch.tensor(ids)


class PickleBasedDataloader:

    def __init__(self, filePaths, metaData, batchSize, doShuffle=True, maxNumSamples=None, maxFilesInMemory=1,
                 ddpWorldSize=None, ddpRank=None, verbose=True):
        self.verbose = verbose
        self.batchSize = batchSize
        self.doShuffle = doShuffle
        self.metaData = metaData
        self.maxNumSamples = maxNumSamples
        self.maxFilesInMemory = maxFilesInMemory
        if maxFilesInMemory > 1:
            raise NotImplementedError("WARNING: maxFilesInMemory > 1. Data Leakage between Epochs will occur!")
        self.filePath2StartEndSamples, self.numSamples, self.numBatches = self.createRelevantMetadata(filePaths,
                                                                                                      metaData,
                                                                                                      maxNumSamples)

        # Data Distributed Parallel Support
        self.ddpRank = ddpRank
        self.ddpWorldSize = ddpWorldSize
        if (self.ddpWorldSize is not None and ddpRank is not None):
            print("Using data distributed data loader: World size {} and ddpRank {}".format(ddpWorldSize, ddpRank))
            self.filePath2StartEndSamples, self.numSamples, self.numBatches = self.createDataDistributedRelevantMetadata(
                ddpWorldSize, ddpRank)

        if (maxNumSamples is not None and batchSize > maxNumSamples):
            print("Lowering batch size {} so match max number of samples {}".format(batchSize, maxNumSamples))
            self.batchSize = maxNumSamples

        self.dataStream = []
        self.fileOrder, self.currentFiles, self.dataLoader = self.initIterator()

    def createRelevantMetadata(self, filePaths, metaData, maxNumSamples=None):
        filePath2numSamples, numSamples = {}, 0
        for f in filePaths:
            try:
                key = f.split('/')[-1]
                fileSamples = metaData[key]
                if (maxNumSamples is not None):
                    fileSamples = min(fileSamples, maxNumSamples - numSamples)
                    filePath2numSamples[f] = (0, fileSamples)
                    numSamples += fileSamples
                    if (numSamples >= maxNumSamples):
                        break
                else:
                    filePath2numSamples[f] = (0, fileSamples)
                    numSamples += fileSamples
            except:
                print("Warning did not find meta data for file:", f)

        numBatches = int(numSamples / self.batchSize)
        return filePath2numSamples, numSamples, numBatches

    def createDataDistributedRelevantMetadata(self, ddpWorldSize=None, ddpRank=None):
        localFilePath2StartEndSamples = {}
        for fPath, (start, end) in self.filePath2StartEndSamples.items():
            totNumsamples = end
            chunkMinSize = int(np.floor(totNumsamples / ddpWorldSize))
            rest = int(totNumsamples % ddpWorldSize)
            chunkSizes = [chunkMinSize for _ in range(ddpWorldSize)]
            for i in range(rest):  # Distribute the rest over the R first chunks
                chunkSizes[i] += 1

            # Calculate chunk start-end points
            startCounter = 0
            chunkStartEnds = []
            for size in chunkSizes:
                chunkStartEnds.append((startCounter, startCounter + size))
                startCounter += size

            # print("Chunk sizes", chunkSizes)
            # print("Chunk Start-Ends", chunkStartEnds)
            # If we got any local relevant files we extract the relevant start-end chunk
            if (chunkSizes[ddpRank] > 0):
                localFilePath2StartEndSamples[fPath] = chunkStartEnds[ddpRank]

        # Calculate new totSamples and totBatches
        numSamples = 0
        for fPath, (start, end) in localFilePath2StartEndSamples.items():
            numSamples += end - start
        numBatches = int(numSamples / self.batchSize)

        # print("Local num samples", numSamples)
        # print("Local Num batches", numBatches)
        return localFilePath2StartEndSamples, numSamples, numBatches

    def initIterator(self):
        self.fileOrder = list(self.filePath2StartEndSamples.items())
        self.fileCounter = 0
        if (self.doShuffle):
            np.random.shuffle(self.fileOrder)

        # Load the first files into the memory and populate the data stream
        self.currentFiles = []
        for _ in range(self.maxFilesInMemory):
            self.loadNewData()

        dataLoader = self.createDataLoaderChunk()
        return self.fileOrder, self.currentFiles, dataLoader

    def loadNewData(self):
        fPath, (start, end) = self.fileOrder[self.fileCounter]
        self.fileCounter = (self.fileCounter + 1) % len(self.fileOrder)
        for f, data in self.currentFiles:
            if (f == fPath):
                if (self.verbose):
                    print("Reusing cached file:", fPath)
                newData = data
                break
        else:
            if (self.verbose):
                print("Loading dataset file:", fPath)
            with open(fPath, 'rb') as fp:
                newData = pickle.load(fp)
                newData = newData[start:end]  # Only load data that's relevant

        # Add the new data to the current files tracker and remove old ones
        self.currentFiles.append((fPath, newData))
        if (len(self.currentFiles) > self.maxFilesInMemory):
            self.currentFiles = self.currentFiles[-self.maxFilesInMemory:]

        # Add the new data to the data stream
        self.dataStream.extend(newData)
        if (self.doShuffle):
            np.random.shuffle(self.dataStream)

    def createDataLoaderChunk(self):
        # print("Num current files", len(self.currentFiles), "Num samples", len(self.dataStream))
        # print(len(self.currentFiles[0][1]))
        currentSize = len(self.currentFiles[0][1])
        # print("Current Size 1:", currentSize)
        # print("Batch Size", self.batchSize)
        currentSize = int(currentSize / self.batchSize) * self.batchSize  # Make sure our dataloader has full batches
        # print("Current Size 2:", currentSize)
        while (currentSize > len(self.dataStream)):
            self.loadNewData()
        # print("Current Size 3:", currentSize)

        # Pop samples from the datastream
        # print(self.fileCounter, len(self.fileOrder))
        # print("Pre", len(self.dataStream))
        chunkIDs = self.dataStream[:currentSize]
        self.dataStream = self.dataStream[currentSize:]
        # print("Post", len(self.dataStream))

        print("ChunkIDs", len(chunkIDs), "CurrentSize", currentSize, "Datastream", len(self.dataStream))
        return iter(torch.utils.data.DataLoader(IDsDataset(chunkIDs), batch_size=self.batchSize, shuffle=self.doShuffle,
                                                num_workers=0, pin_memory=False))

    def __iter__(self):
        self.batchCounter = 0
        return self

    def __len__(self):
        return self.numBatches

    def __next__(self):
        self.batchCounter += 1
        if (self.batchCounter > self.numBatches):
            raise StopIteration

        try:
            batchIDs = next(self.dataLoader)
            return batchIDs
        except StopIteration:  # The current data loader is out of data, so we load a new one
            self.dataLoader = self.createDataLoaderChunk()
            return next(self.dataLoader)


def main():
    from DataManagers import DataUtils
    # dataloader = DataUtils.loadPreTokenizedDataloader(16, 'OpenWeb', 16384)

    device = 'cuda:0'
    batchSize = 4
    datasetName = 'Wikitext_gpt2'
    vocabSize = 50257
    dataLoaders = [
        DataUtils.loadPreTokenizedDataloader(batchSize, datasetName,
                                             vocabSize,
                                             DataUtils.DatasetSplit.VALID,
                                             maxSeqLen=128,
                                             maxNumSamples=500)
        for _ in range(4)
    ]

    for i in tqdm.tqdm(range(100)):
        for dataset in dataLoaders:
            for bIDs in dataset:
                bIDs = bIDs.to(device)
