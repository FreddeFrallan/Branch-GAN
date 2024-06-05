import numpy as np
import wandb
import torch
import time
import os


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def entropy(probs, epsilon=1e-8):
    return -torch.sum(probs * torch.log(probs + epsilon), dim=-1)


def initLog(configData, project="non-residual-gann", entity="hackerz", runName='Mini-{}', disabled=False,
            addTimeStamptToName=False):
    timestamp = time.time()
    configData['ModelID'] = timestamp
    if (addTimeStamptToName):
        runName = runName.format(timestamp)

    if (disabled):
        wandb.init(name=runName, config=configData, project=project, entity=entity, mode='disabled')
    else:
        wandb.init(name=runName, config=configData, project=project, entity=entity)

    return runName, timestamp


def finishWandb():
    wandb.finish()


def logValues(data, commit=True):
    wandb.log(data, commit=commit)


def logText(data, columns, name, commit=True):
    logValues({name: wandb.Table(data=data, columns=columns)}, commit=commit)


def saveGannSetup(model, runName, timeMarker, discOptimizer=None, genOptimizer=None, mainSavePath='',
                  datasetConf=None, trainingConf=None, tokenizer=None, saveDisc=True,
                  saveGen=True, saveOptimizers=True):
    print("Saving model:", runName)
    folderName = runName.replace('/', '-')
    mainSaveFolder = os.path.join(mainSavePath, folderName)
    if (os.path.exists(mainSaveFolder) == False):
        os.makedirs(mainSaveFolder)

    # Save configs if they're yet to be created
    architectureConfPath = os.path.join(mainSaveFolder, 'Architecture-Config.json')
    discConfPath = os.path.join(mainSaveFolder, 'Discriminator-Config.json')
    # trainingConfPath = os.path.join(mainSaveFolder, 'Training-Config.json')
    datasetConfPath = os.path.join(mainSaveFolder, 'Dataset-Config.json')
    genConfPath = os.path.join(mainSaveFolder, 'Generator-Config.json')
    for savePath, config in [(discConfPath, model.discriminatorConfig), (genConfPath, model.generatorConfig),
                             (datasetConfPath, datasetConf),
                             ]:
        if (os.path.exists(savePath) == False and config is not None):
            config.to_json_file(savePath)

    if (tokenizer is not None):
        tokenizer.save(os.path.join(mainSaveFolder, 'Tokenizer.json'))

    currentSaveFolder = os.path.join(mainSaveFolder, timeMarker)
    generatorFolder = os.path.join(currentSaveFolder, 'Generator')
    discriminatorFolder = os.path.join(currentSaveFolder, 'Discriminator')
    for f in [currentSaveFolder, generatorFolder, discriminatorFolder]:
        if (os.path.exists(f) == False):
            os.makedirs(f)

    if (saveGen):
        generatorWeightsPath = os.path.join(generatorFolder, 'Generator.pt')
        torch.save(model.generator.state_dict(), generatorWeightsPath)
    if (saveDisc):
        discriminatorWeightsPath = os.path.join(discriminatorFolder, 'Discriminator.pt')
        torch.save(model.discriminator.state_dict(), discriminatorWeightsPath)

    # Save Optimizers
    if(saveOptimizers):
        for optimizer, saveFolder, doSave in [(discOptimizer, discriminatorFolder, saveDisc),
                                              (genOptimizer, generatorFolder, saveGen)]:
            if (doSave and optimizer is not None):
                optimizer.saveOptimizer(os.path.join(saveFolder, 'Optimizer.pt'))


def _loadStateDisctFromPotentialDistributedModel(path):
    stateDisct = torch.load(path, map_location=torch.device('cpu'))
    return {k.replace('module.', ''): v for k, v in stateDisct.items()}


def getAvailablCheckpointsFromModelFolder(modelFolder):
    availableSubFolders = []
    f = lambda x, y: os.path.join(x, y)
    for file in os.listdir(modelFolder):
        if (os.path.isdir(f(modelFolder, file))):
            generatorPath = f(f(modelFolder, file), 'Generator/Generator.pt')
            discriminatorPath = f(f(modelFolder, file), 'Discriminator/Discriminator.pt')
            if (os.path.exists(discriminatorPath) and os.path.exists(generatorPath)):
                availableSubFolders.append(file)

    def extractSortValue(name):
        if ('Epoch-' in name):
            return float(name.split('-')[-1])
        return 999999

    return list(sorted(availableSubFolders, key=lambda x: extractSortValue(x)))


class ModelSaveFrequencyTracker:

    def __init__(self, frequency, datasetSize, enabled=True, maxEpoch=2, epochSaveFrequency=1):
        self.enabled = enabled
        self.frequency = frequency
        self.datasetSize = datasetSize
        self.maxEpoch = maxEpoch
        self.epochSaveFrequency = epochSaveFrequency

    def initEpoch(self):
        if (self.enabled == False or self.frequency is None):
            self.nextSaveThreshold = None
        else:
            self.nextSaveThreshold = self.frequency * self.datasetSize
            print("Next save threshold", self.nextSaveThreshold)

    def shouldModelBeSaved(self, batchIDx, epoch):
        if (self.enabled == False or self.nextSaveThreshold is None):
            return False, None

        if (self.maxEpoch is not None and epoch >= self.maxEpoch):
            return False, None

        if (batchIDx >= self.nextSaveThreshold):
            epochDecimals = np.around(self.nextSaveThreshold / self.datasetSize, 4)
            saveName = "Epoch-{}".format(epoch + epochDecimals)

            self.nextSaveThreshold += self.frequency * self.datasetSize
            print("Next save threshold", self.nextSaveThreshold)
            return True, saveName
        else:
            return False, None

    def shouldModelBeSavedAtThisEpoch(self, finishedEpoch):
        return finishedEpoch % self.epochSaveFrequency == 0
