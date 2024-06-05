from Training import GanUpdateSchedule
from Configs import ReinforceLabelMode
from Configs.BaseConfig import Config
import numpy as np
import torch
import copy
import enum


class DatasetSize(enum.IntEnum):
    ALL = 100
    THREEQUARTER = 75
    HALF = 50
    ONEQUARTER = 25
    ONETENTH = 10
    ONEPERCENT = 1
    ONETENTPERCENT = 0.1
    ONEHUNDERETHPERCENT = 0.01


class DatasetConfig(Config):

    def __init__(self, datasetPath, maxNumSamples=None):
        self.datasetPath = datasetPath
        self.maxNumSamples = maxNumSamples


class OptimizerConfig(Config):

    def __init__(self, lossKey='loss', name='adam', lr=0.0001, gradAccum=1, params=None, schedulerType=None,
                 schedulerArgs=None, useZeroOptimizer=False):
        self.lossKey = lossKey
        self.name = name
        self.lr = lr

        self.gradAccum = gradAccum
        self.params = params if params is not None else {}

        self.schedulerArgs = schedulerArgs
        self.schedulerType = schedulerType
        self.useZeroOptimizer = useZeroOptimizer


class TrainingConfig(Config):

    def __init__(self, device='cuda:0', batchSize=128, validationBatchSize=None, numEpochs=4,
                 numNegativeUpdateAlterations=8,
                 trainingSamplingConf=None, validationSamplingConf=None,
                 mleLossWeight=0, rlLossWeight=1,
                 discPredLossWeight=1, discValueHeadLossWeight=0.25,
                 runName='TestExperiment',
                 genOptimConfig=None, discOptimConfig=None,
                 updatesPerEvaluation=100,
                 doLogData=True, displayProgression=False, saveModel=True,
                 generatorTeacherForcing=True,
                 genDiscUpdateSchedule=GanUpdateSchedule.AlwaysUpdateSchedule(),
                 genStartWeights=None, discStartWeights=None, setupStartWeights=None,
                 genOptimizerStartWeights=None, discOptimizerStartWeights=None,
                 trainingSetSize=DatasetSize.ALL,
                 testSetSize=DatasetSize.ONETENTH,
                 additionalModelSaveFreq=None,
                 reinforcementLearningConfig=ReinforceLabelMode.ReinforcementLearningConfig(),
                 generationEvaluationConfig=None,
                 devicesPerConfig=1,
                 mixedPrecision=False,
                 saveOptimizers=True,
                 depthSchedule=None,
                 **kwargs,
                 ):
        self.kwargs = kwargs
        self.device = device
        self.mixedPrecision = mixedPrecision
        self.devicesPerConfig = devicesPerConfig
        self.kwargs['mixedPrecision'] = mixedPrecision
        self.saveOptimizers = saveOptimizers

        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.validationBatchSize = validationBatchSize

        self.generationEvaluationConfig = generationEvaluationConfig
        self.reinforcementLearningConfig = reinforcementLearningConfig
        self.numNegativeUpdateAlterations = numNegativeUpdateAlterations

        self.mleLossWeight = mleLossWeight
        self.rlLossWeight = rlLossWeight
        self.discValueHeadLossWeight = discValueHeadLossWeight
        self.discPredLossWeight = discPredLossWeight

        self.trainingSamplingConf = trainingSamplingConf
        self.validationSamplingConf = validationSamplingConf

        self.genOptimConfig = genOptimConfig if genOptimConfig is not None else OptimizerConfig()
        self.discOptimConfig = discOptimConfig if discOptimConfig is not None else OptimizerConfig()

        if discOptimConfig is None:
            gradAccum = 1
        else:
            gradAccum = discOptimConfig[0].gradAccum if type(discOptimConfig) == list else discOptimConfig.gradAccum

        self.updatesPerEvaluation = updatesPerEvaluation * gradAccum
        self.additionalModelSaveFreq = additionalModelSaveFreq
        self.displayProgression = displayProgression
        self.saveModel = saveModel
        self.doLogData = doLogData
        self.runName = runName

        self.generatorTeacherForcing = generatorTeacherForcing
        self.genDiscUpdateSchedule = genDiscUpdateSchedule
        self.depthSchedule = depthSchedule

        self.genStartWeights = genStartWeights
        self.discStartWeights = discStartWeights
        self.setupStartWeights = setupStartWeights
        self.genOptimizerStartWeights = genOptimizerStartWeights
        self.discOptimizerStartWeights = discOptimizerStartWeights

        self.trainingSetSize = trainingSetSize
        self.testSetSize = testSetSize

    def to_dict(self):
        data = copy.deepcopy(self.__dict__)

        def representOptimConfig(config):
            if (type(config) == list):
                return [c.to_dict() for c in config]
            else:
                return config.to_dict()

        data['genOptimConfig'] = representOptimConfig(self.genOptimConfig)
        data['discOptimConfig'] = representOptimConfig(self.discOptimConfig)
        return data


class SamplingConfig(Config):

    def __init__(self, samplingRules, discDepthLossSchedule=None, includeAllMultiStepSamplesForDisc=True,
                 separateInputAltLoss=True, inputLossWeight=1, discpDepthLossWeight=None, numSampleSequences=None,
                 sparseIndex=None):
        if (type(samplingRules) != list):
            self.rules = [samplingRules]
        else:
            self.rules = samplingRules

        if (len(self.rules) > 1):
            self.numAlterations = int(np.prod([r.numAlterations for r in self.rules]))
        else:
            self.numAlterations = self.rules[0].numAlterations

        self.multiStepSampling = True
        self.includeAllMultiStepSamplesForDisc = includeAllMultiStepSamplesForDisc
        self.discDepthLossSchedule = discDepthLossSchedule
        self.discpDepthLossWeight = discpDepthLossWeight
        self.separateInputAltLoss = separateInputAltLoss
        self.inputLossWeight = inputLossWeight
        self.numSampleSequences = numSampleSequences

        # Convert sparse index to tensor if it is not already
        if (sparseIndex is not None and type(sparseIndex) != torch.Tensor):
            sparseIndex = torch.tensor(sparseIndex)
        self.sparseIndex = sparseIndex
        self._sampleIndex = None

    def getNextSampleConf(self):
        return self

    def getSparseIndex(self, ids, device):
        if (self.sparseIndex is not None):
            return self.sparseIndex
        if (self.numSampleSequences is None or self.numSampleSequences == ids.shape[1]):
            return None

        self._sampleIndex = np.random.choice(ids.shape[1] - 1, self.numSampleSequences, replace=False)
        self._sampleIndex = torch.tensor(np.sort(self._sampleIndex), device=device)

        return self._sampleIndex

    def to_dict(self):
        data = {'IncludeAllMultiStepSamplesForDisc': self.includeAllMultiStepSamplesForDisc,
                'DiscDepthLossSchedule': str(
                    self.discDepthLossSchedule.to_dict()) if self.discDepthLossSchedule is not None else None,
                'separateInputAltLoss': self.separateInputAltLoss,
                'inputLossWeight': self.inputLossWeight,
                'numSampleSequences': self.numSampleSequences,
                }
        for i, rule in enumerate(self.rules):
            data['Rule-{}'.format(i)] = rule.to_dict()
        return data

    def __repr__(self):
        return str(self.to_dict())


class RoundRobinSamplingConfig(Config):

    def __init__(self, samplingConfs):
        self.samplingConfs = samplingConfs
        self.sampleCounter = 0

    def getNextSampleConf(self):
        self.sampleCounter += 1
        self.sampleCounter = self.sampleCounter % len(self.samplingConfs)
        return self.samplingConfs[self.sampleCounter]

    def to_dict(self):
        data = {}
        for i, conf in enumerate(self.samplingConfs):
            data['Conf-{}'.format(i)] = conf.to_dict()
        return data

    def __repr__(self):
        return str(self.to_dict())


class TextGenerationConfig(Config):

    def __init__(self, genSettings=None, maxNumSamples=128, contextSize=128, maxGenerationLength=128):
        self.maxGenerationLength = maxGenerationLength
        self.maxNumSamples = maxNumSamples
        self.contextSize = contextSize

        if (genSettings is None):
            self.genSettings = {
                'Greedy-1.0':
                    {
                        'temperature': 1.0,
                        'num_beams': 1,
                        'do_sample': False,
                        'num_return_sequences': 1
                    },
                'Sample-1.0':
                    {
                        'temperature': 1.0,
                        'num_beams': 1,
                        'do_sample': True,
                        'num_return_sequences': 1
                    },
            }
        else:
            assert type(genSettings) == dict
            assert len(genSettings) > 0
            self.genSettings = genSettings

    def to_dict(self):
        return {'maxGenerationLength': self.maxGenerationLength, 'maxNumSamples': self.maxNumSamples,
                'contextSize': self.contextSize, 'genSettings': self.genSettings}
