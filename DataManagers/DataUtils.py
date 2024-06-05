from DataManagers import PickleBasedDataloader
from pathlib import Path
import json
import os


def _makeParentDirs(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def getAllMatchingDatasetFiles(mainPath, dataFileExtension='.pkl', metaDataFileExtension='.json'):
    dataFiles, metaFile = [], None

    for root, dirs, files in os.walk(mainPath):
        for file in files:
            if file.endswith(dataFileExtension):
                dataFiles.append(os.path.join(root, file))
            elif file.endswith(metaDataFileExtension):
                metaFile = os.path.join(root, file)

    if metaFile is None:
        raise ValueError('No meta file found in {}'.format(mainPath))

    with open(metaFile, 'r') as fp:
        metaData = json.load(fp)

    return dataFiles, metaData


def loadPreTokenizedDataloader(batchSize, datasetPath, doShuffle=True, maxNumSamples=None, ddpWorldSize=None,
                               ddpRank=None, verbose=True):
    dataFiles, metaData = getAllMatchingDatasetFiles(datasetPath)
    print("Total number of available files: {}".format(len(dataFiles)))

    return PickleBasedDataloader.PickleBasedDataloader(dataFiles, metaData, batchSize, doShuffle=doShuffle,
                                                       maxNumSamples=maxNumSamples, ddpWorldSize=ddpWorldSize,
                                                       ddpRank=ddpRank, verbose=verbose)
