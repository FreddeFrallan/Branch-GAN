from BranchPredictions import MultiStepPrediction, SamplingStrategies
from Training.Metrics import GeneratorMetrics, DiscriminatorMetrics
from Models.Losses import DiscriminatorLoss, GeneratorLosses
from Models import Discriminator, Generator
from Configs import ReinforceLabelMode
import torch
import math


class GannSetup(torch.nn.Module):

    def __init__(self, generatorConfig, discriminatorConfig, mleLossWeight=1, rlLossWeight=1,
                 discPredLossWeight=1, discValueHeadLossWeight=0.25, device='cpu',
                 generatorDevice=None, discriminatorDevice=None,
                 generatorTeacherForcingInput=True,
                 samplingConf=None,
                 **modelCreationConfigs):
        super().__init__()
        self.device = device
        self.generatorConfig = generatorConfig
        self.discriminatorConfig = discriminatorConfig
        self.generatorDevice = generatorDevice if generatorDevice is not None else device
        self.discriminatorDevice = discriminatorDevice if discriminatorDevice is not None else device
        print("Setting Non-Res Setup. Generator Device: {} Discriminator Device: {}".format(self.generatorDevice,
                                                                                            self.discriminatorDevice))

        self.generator, self.discriminator = self.createGeneratorAndDiscriminator(generatorConfig, discriminatorConfig)

        # Generator Loss Weights
        self.generatorMleLossWeight = mleLossWeight
        self.generatorRlLossWeight = rlLossWeight

        # Discriminator Loss Weights
        self.discValueHeadLossWeight = discValueHeadLossWeight
        self.discPredLossWeight = discPredLossWeight

        # SamplingStrategies
        self.samplingConf = samplingConf

        # Teacher forcing settings
        self.generatorTeacherForcingInput = generatorTeacherForcingInput

        self.discriminatorLoss = torch.nn.functional.binary_cross_entropy
        self.discValueHeadLoss = torch.nn.functional.binary_cross_entropy
        self.valueHeadLoss = torch.nn.MSELoss()

        if (modelCreationConfigs.get('mixedPrecision', False)):
            self.lossScaler = torch.cuda.amp.GradScaler()

    def createGeneratorAndDiscriminator(self, generatorConfig, discriminatorConfig):
        discriminator = Discriminator.NonResidualDiscriminatorWithDualValueHeads(discriminatorConfig)
        generator = Generator.NonResidualGenerator(generatorConfig)
        return generator, discriminator

    def generativeForwardPass(self, inputIDs, samplingConf):
        samplingConf = self.samplingConf if samplingConf is None else samplingConf

        output = self.generator.forward(
            {'input_ids': inputIDs, 'output_hidden_states': True, 'returnLmLogits': True})
        alterationIDs, maskedProbs = SamplingStrategies.sampleWithSamplingRule(output.classProbs, inputIDs,
                                                                               samplingConf.rules,
                                                                               blockSamplingOfLabels=False,
                                                                               includeFinalTimeStep=True
                                                                               )

        idsPadding = torch.ones((inputIDs.shape[0], 1)).to(inputIDs.device).type(inputIDs.dtype)
        extendedInputIDs = torch.cat((inputIDs, idsPadding), dim=-1)

        f = lambda x: torch.permute(x, dims=(0, 2, 1))
        alterationGenProbs = torch.gather(output.classProbs, dim=-1, index=f(alterationIDs)[:, 1:])
        inputPreds, altPreds = self.discriminator.discForward(extendedInputIDs, alterationIDs)

        return output.classProbs, f(alterationGenProbs), alterationIDs, inputPreds, altPreds

    def createDiscAlterationLossWeights(self, inputIDs, alternationNodes, depthSchedule=None, customDepthWeights=None):
        discSeqDepths = []
        for n in alternationNodes:
            discSeqDepths.append(n.getPaddedPosIDs())

        # Mask out depths that are beyond max seq length
        discSeqDepths = torch.stack(discSeqDepths).to(inputIDs.device)
        discSeqDepthMask = torch.less_equal(discSeqDepths, inputIDs.shape[-1] - 1).float()

        if (customDepthWeights is not None):
            customDiscDepthMask = customDepthWeights.calculateLossWeights(alternationNodes, inputIDs)
            discSeqDepthMask = discSeqDepthMask * customDiscDepthMask

        if (depthSchedule is not None):
            depthWeights = depthSchedule.calculateLossWeights(alternationNodes, inputIDs)
            return discSeqDepthMask, depthWeights

        return discSeqDepthMask, None

    def _addDiscValueHeadsToMultiStepPredictions(self, inputIDs, multiStepRoot, alterationNodes, forwardData):
        f = lambda x: torch.stack(x, dim=0)
        temp = {'txtInputPreds': multiStepRoot.valueHeadData['InputValue'],
                'txtAlterationsPreds': multiStepRoot.valueHeadData['AlterationsValue'],
                'altInputPreds': f([n.valueHeadData['InputValue'] for n in alterationNodes]),
                'altAlterationsPreds': f([n.valueHeadData['AlterationsValue'] for n in alterationNodes])
                }
        forwardData['DiscValueHeads'] = temp
        forwardData['SubNodeStructure'] = MultiStepPrediction.getSubNodeListTreeStructure(multiStepRoot,
                                                                                          alterationNodes)
        return forwardData

    def _multiStepForwardPass(self, inputIDs, samplingConf, collectTreeDepthData=True):
        multiStepRoot, firstPred, probs, firstMaskedProbs = MultiStepPrediction.multiStepGenForward(
            self.generator, inputIDs,
            samplingConf.rules,
            self.generatorDevice,
            sparseIndex=samplingConf.getSparseIndex(inputIDs, self.generatorDevice),
        )

        # Get the multistep predictions and potential loss weights
        alterationNodes = multiStepRoot.getSubNodes() if samplingConf.includeAllMultiStepSamplesForDisc else multiStepRoot.getLeaves()
        depthMask, discDepthWeights = self.createDiscAlterationLossWeights(inputIDs, alterationNodes,
                                                                           depthSchedule=samplingConf.discDepthLossSchedule,
                                                                           customDepthWeights=samplingConf.discpDepthLossWeight)

        # Pad the alteration probs
        alterationProbs = torch.stack([n.getPaddedAlterationProbs() for n in alterationNodes], dim=1)

        MultiStepPrediction.multiStepDiscForward(self.discriminator, multiStepRoot, self.discriminatorDevice,
                                                 dualValueHeads=True)
        discAlterationPreds = [n.discPreds for n in alterationNodes]
        for i, n in enumerate(alterationNodes):
            print(i, n.discPreds.prediction.shape)

        # Collect Tree Node information
        forwardData = {'TreeNodeInfo': [(n.depth, n.isRandom) for n in alterationNodes]} if collectTreeDepthData else {}
        forwardData['AlterationProbs'] = alterationProbs

        # If our discriminator has value head data, we add it to the forward pass data
        forwardData = self._addDiscValueHeadsToMultiStepPredictions(inputIDs, multiStepRoot, alterationNodes,
                                                                    forwardData)

        discInputPred = multiStepRoot.discPreds
        MultiStepPrediction.clearTree(multiStepRoot, clearAll=True)

        genData = (firstPred.logits, probs, firstMaskedProbs)
        discData = (discInputPred, discAlterationPreds, depthMask, discDepthWeights)
        return inputIDs, genData, discData, forwardData

    def forwardPass(self, inputIDs, samplingConf, **kwargs):
        samplingConf = self.samplingConf if samplingConf is None else samplingConf
        if (kwargs.get('mixedPrecision', False)):
            with torch.cuda.amp.autocast():
                return self._multiStepForwardPass(inputIDs, samplingConf)
        return self._multiStepForwardPass(inputIDs, samplingConf)

    def _getStackedAndPaddedDiscAlterationPreds(self, discAlterationPreds, discInputPreds):
        inputPrediction = discInputPreds.prediction
        if (type(discAlterationPreds) == list):
            discAlterationScores = torch.stack([p.prediction for p in discAlterationPreds], dim=1)
            for i, p in enumerate(discAlterationPreds):
                print(i, p.prediction.shape)
            print(inputPrediction.shape, discAlterationScores.shape)
            if (inputPrediction.shape[1] > discAlterationScores.shape[1]):
                pad = torch.zeros((inputPrediction.shape[0], discAlterationScores.shape[1],
                                   inputPrediction.shape[1] - discAlterationScores.shape[2],
                                   ),
                                  device=discAlterationScores.device)
                discAlterationScores = torch.cat([discAlterationScores, pad], dim=-1)
        else:
            discAlterationScores = discAlterationPreds.prediction

        return discAlterationScores

    def calcDiscMetrics(self, discInputPreds, discAlterationPreds, forwardPassData):
        discInputScores = discInputPreds.prediction
        if (type(discAlterationPreds) == list):
            discAlterationScores = torch.stack([p.prediction for p in discAlterationPreds], dim=1)
        else:
            discAlterationScores = discAlterationPreds.prediction

        discMetrics = DiscriminatorMetrics.calcDiscriminatorMetrics(discInputScores, discAlterationScores,
                                                                    forwardPassData)

        return discMetrics

    def _calcDiscLoss(self, discInputPreds, discAlterationPreds, numNegativeUpdateSamples, depthMask,
                      discDepthWeights, forwardPassData, **kwargs):
        inputPrediction = discInputPreds.prediction
        discAlterationScores = self._getStackedAndPaddedDiscAlterationPreds(discAlterationPreds, discInputPreds)
        return DiscriminatorLoss.calcDiscriminatorLoss(self, inputPrediction, discAlterationScores,
                                                       numNegativeUpdateSamples, depthMask, discDepthWeights,
                                                       collectTreeDepthData=kwargs.get(
                                                           'collectTreeDepthData',
                                                           False),
                                                       forwardPassData=forwardPassData,
                                                       )

    def _internalTrainForwardPass(self, inputIDs, samplingConf, numNegativeUpdateSamples=1,
                                  **kwargs):
        inputIDs, genData, discData, forwardPassData = self.forwardPass(inputIDs, samplingConf, **kwargs)
        genLogits, genProbs, maskedProbs = genData
        discInputPreds, discAlterationPreds, depthMask, discDepthWeights = discData

        data = forwardPassData['DiscValueHeads']
        txtAlterationsPreds, altAlterationsPreds = data['txtAlterationsPreds'], data['altAlterationsPreds']

        # Pad the altPredictions if needed
        if (txtAlterationsPreds.shape[1] > altAlterationsPreds.shape[2]):
            b, a, s = altAlterationsPreds.shape[0], altAlterationsPreds.shape[1], altAlterationsPreds.shape[2]
            pad = torch.zeros(b, a, txtAlterationsPreds.shape[1] - s, 1).to(altAlterationsPreds.device)
            altAlterationsPreds = torch.cat([altAlterationsPreds, pad], dim=2)
            data['altAlterationsPreds'] = altAlterationsPreds

        discInputLoss, discAlterationLoss, discLossInfo = self._calcDiscLoss(discInputPreds, discAlterationPreds,
                                                                             numNegativeUpdateSamples, depthMask,
                                                                             discDepthWeights, forwardPassData,
                                                                             **kwargs)

        if (samplingConf.separateInputAltLoss):
            discPredLoss = (discInputLoss * samplingConf.inputLossWeight + discAlterationLoss) / 2
        else:
            numAlterations = discLossInfo['numAlterationSamples']
            discPredLoss = (discInputLoss * samplingConf.inputLossWeight + (discAlterationLoss * numAlterations)) / (
                    numAlterations + 1)

        discLoss = discPredLoss * self.discPredLossWeight + discLossInfo['ValueHeadLoss'] * self.discValueHeadLossWeight

        # Create loss dicts
        discLosses = {'loss': discLoss, 'predLoss': discPredLoss, 'loss-Input': discInputLoss,
                      'loss-Alterations': discAlterationLoss}

        # Calc generator Loss
        # with torch.no_grad():
        discAlterationScores = self._getStackedAndPaddedDiscAlterationPreds(discAlterationPreds, discInputPreds)

        f = lambda x: x.to(self.generatorDevice) if x is not None else None
        g = lambda x: f(x.detach())
        genLosses = GeneratorLosses.calcGeneratorLoss(self, f(inputIDs), genLogits, genProbs,
                                                      g(discInputPreds.prediction), g(discAlterationScores),
                                                      forwardPassData,
                                                      reinforcementLearningConfig=kwargs.get(
                                                          'reinforcementLearningConfig',
                                                          ReinforceLabelMode.ReinforcementLearningConfig(),
                                                      ),
                                                      depthMask=f(depthMask),
                                                      )

        discLosses['ValueHeadLoss'] = discLossInfo['ValueHeadLoss']

        # Add depth losses to the log. This dictionary can be empty
        for k, v in discLossInfo.items():
            discLosses[k] = v

        # Check if any of the gen losses is NaN, and if so raise an exception
        for k, v in genLosses.items():
            if (type(v) == torch.Tensor):
                if (torch.isnan(v)):
                    raise Exception("Generator loss {} is NaN".format(k))
            else:
                if (math.isnan(v)):
                    raise Exception("Generator loss {} is NaN".format(k))

        return discLosses, genLosses, genData, discData, forwardPassData

    def trainForwardPass(self, inputIDs, samplingConf, numNegativeUpdateSamples=1, **kwargs):
        discLosses, genLosses, genData, discData, forwardPassData = self._internalTrainForwardPass(inputIDs,
                                                                                                   samplingConf,
                                                                                                   numNegativeUpdateSamples,
                                                                                                   **kwargs)
        genLogits, genProbs, maskedProbs, = genData
        discInputPreds, discAlterationPreds, _, _ = discData
        discInputScores = discInputPreds.prediction
        if (type(discAlterationPreds) == list):
            discAlterationScores = torch.stack([p.prediction for p in discAlterationPreds], dim=1)
        else:
            discAlterationScores = discAlterationPreds.prediction

        # Move tensors to generator device
        genInputIDs = inputIDs.to(self.generatorDevice)
        discInputScoresForGenerator = discInputScores.detach().to(self.generatorDevice)
        discAlterationScoresForGenerator = discAlterationScores.detach().to(self.generatorDevice)
        generatorMetrics = GeneratorMetrics.calculateGeneratorMetrics(genInputIDs,
                                                                      genProbs, maskedProbs,
                                                                      discInputScoresForGenerator,
                                                                      discAlterationScoresForGenerator,
                                                                      )

        discMetrics = self.calcDiscMetrics(discInputPreds, discAlterationPreds, forwardPassData)

        return discLosses, genLosses, discMetrics, generatorMetrics

    def trainForwardPassNoMetrics(self, inputIDs, samplingConf, numNegativeUpdateSamples=1, **kwargs):
        discLosses, genLosses, genData, discData, forwardPassData = self._internalTrainForwardPass(inputIDs,
                                                                                                   samplingConf,
                                                                                                   numNegativeUpdateSamples,
                                                                                                   **kwargs)
        return discLosses, genLosses
