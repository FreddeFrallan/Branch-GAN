from Training.Metrics import GeneratedTextsMetric
from Training import GanUpdateSchedule, Utils
from Inference import GenerateTexts
import numpy as np
import torch
import tqdm


def _appendToCollectedData(collectedData, newData, ignoreTrackingKeys=()):
    for i, (k, v) in enumerate(newData.items()):
        if k in ignoreTrackingKeys:
            continue
        if (k not in collectedData):
            collectedData[k] = []
        if (type(v) == list):
            collectedData[k].extend(v)
        elif (type(v) == int):
            collectedData[k].append(v)
        elif (len(v.shape) >= 1):
            collectedData[k].append(v.cpu().numpy())
        else:
            collectedData[k].append(v.item())


def forwardPass(model, inData, sampleConf, numUpdateSamples, stepCounter=None,
                collectedGenData=None, collectedDiscData=None,
                updateGenerator=False, updateDiscriminator=False,
                generatorOptim=None, discriminatorOptim=None,
                ignoreTrackingKeys=()):
    discLosses, genLosses, discMetrics, genMetrics = model.trainForwardPass(inData,
                                                                            sampleConf,
                                                                            numUpdateSamples,
                                                                            collectData=True
                                                                            )
    if (updateGenerator):
        generatorOptim.performUpdateStep(genLosses, stepCounter)
    if (updateDiscriminator):
        discriminatorOptim.performUpdateStep(discLosses, stepCounter)

    # Add losses to the metrics
    for origin, target in [(discLosses, discMetrics), (genLosses, genMetrics)]:
        for k, v in origin.items():
            if (k not in target):
                target[k] = v

    # Add metric and losses to the collected data
    collectedDiscData = collectedDiscData if collectedDiscData is not None else {}
    collectedGenData = collectedGenData if collectedGenData is not None else {}
    _appendToCollectedData(collectedDiscData, discMetrics, ignoreTrackingKeys)
    _appendToCollectedData(collectedGenData, genMetrics, ignoreTrackingKeys)

    return collectedDiscData, collectedGenData


def logResults(validationResult, genData, discData, trainDataset, epochCounter,
               batchCounter, generatorOptim, discriminatorOptim, trainingConf):
    # Modify this function to suit your logging purposes

    def logMetrics(data, prefix, aggrigation=np.mean, selectedKeys=None):
        targetKeys = selectedKeys if selectedKeys is not None else data.keys()
        logData = {"{}-{}".format(prefix, k): aggrigation(data[k]) for k in targetKeys}
        print(logData)

    for name, (valDiscData, valGenData) in validationResult.items():
        logMetrics(valDiscData, 'Val-{}_Discriminator'.format(name))
        logMetrics(valGenData, 'Val-{}_Generator'.format(name))

    if (discriminatorOptim.scheduler is not None):
        lr = discriminatorOptim.scheduler.get_last_lr()[0]
        discData['lr'] = lr
    if (generatorOptim.scheduler is not None):
        lr = generatorOptim.scheduler.get_last_lr()[0]
        discData['genData'] = lr
    if (trainingConf is not None and trainingConf.depthSchedule is not None):
        genData['generationDepth'] = trainingConf.depthSchedule.currentDepth

    logMetrics({'Epoch': epochCounter + (batchCounter / len(trainDataset))}, '')
    logMetrics(discData, 'Discriminator')
    logMetrics(genData, 'Generator')



def validateModel(model, testData, samplingStrategies={}, displayProgress=True):
    if (testData is None):
        return {}

    with torch.no_grad():
        results = {}
        for sampleName, conf in samplingStrategies.items():
            discData, genData = {}, {}
            for i, bIDs in enumerate(tqdm.tqdm(testData, 'Evaluating Model-{}'.format(sampleName),
                                               disable=not displayProgress,
                                               # disable=True
                                               )):
                discData, genData = forwardPass(model, bIDs,
                                                conf.getNextSampleConf(),
                                                numUpdateSamples=1,
                                                collectedGenData=genData,
                                                collectedDiscData=discData,
                                                )

                results[sampleName] = (discData, genData)

    return results


def efficentTrainingLoop(model, trainDataset, testData,
                         numEpochs,
                         generatorOptim, discriminatorOptim,
                         trainingSamplingConf, numUpdateSamples, updatesPerEvaluation,
                         validationSamplingConf,
                         doLogData=True, displayProgression=False,
                         saveModel=True, saveName='SavedModel',
                         genDiscUpdateSchedule=GanUpdateSchedule.AlwaysUpdateSchedule(),
                         datasetConf=None, trainingConf=None,
                         **kwargs
                         ):
    modelSaveTracker = Utils.ModelSaveFrequencyTracker(kwargs.get('saveFreq', None),
                                                       len(trainDataset), enabled=saveModel,
                                                       epochSaveFrequency=kwargs.get(
                                                           'epochSaveFreq', None))
    print("Starting training loop")

    stepCounter, epochCounter = 0, 0
    for e in range(int(numEpochs)):
        discData, genData, combinedData = {}, {}, {}
        modelSaveTracker.initEpoch()

        for i, bIDs in enumerate(
                tqdm.tqdm(trainDataset, desc='Epoch: ' + str(epochCounter), disable=not displayProgression)
        ):
            updateDisc, updateGen = genDiscUpdateSchedule.getGenDiscUpdates(stepCounter, epochCounter, generatorOptim,
                                                                            discriminatorOptim)

            if (trainingConf is not None and trainingConf.depthSchedule is not None):
                trainingConf.depthSchedule.updateSchedule(epochCounter, i, stepCounter)
                sampleConf = trainingConf.depthSchedule.setSampleConfDepth(trainingSamplingConf.getNextSampleConf())
            else:
                sampleConf = trainingSamplingConf.getNextSampleConf()

            discLosses, genLosses = model.trainForwardPassNoMetrics(bIDs, sampleConf,
                                                                    numUpdateSamples,
                                                                    collectTreeDepthData=True,
                                                                    UpdateGen=updateGen,
                                                                    reinforcementLearningConfig=trainingConf.reinforcementLearningConfig,
                                                                    **kwargs)

            if (updateGen):
                generatorOptim.performUpdateStep(genLosses, stepCounter)

            if (updateDisc):
                discriminatorOptim.performUpdateStep(discLosses, stepCounter)

            # Log the training losses
            _appendToCollectedData(discData, discLosses)
            _appendToCollectedData(genData, genLosses)

            stepCounter += 1
            if (stepCounter % updatesPerEvaluation == 1):  # Equals 1 because we want an early evaluation
                validationResult = validateModel(model, testData,
                                                 samplingStrategies=validationSamplingConf,
                                                 displayProgress=displayProgression)

                if (doLogData):
                    logResults(validationResult, genData, discData, trainDataset,
                               epochCounter, i, generatorOptim, discriminatorOptim, trainingConf)

                discData, genData, combinedData = {}, {}, {}
                if (saveModel):
                    Utils.saveGannSetup(model, saveName, 'Current', discriminatorOptim,
                                        generatorOptim, datasetConf=datasetConf,
                                        trainingConf=trainingConf,
                                        saveGen=genDiscUpdateSchedule.saveGen, saveDisc=genDiscUpdateSchedule.saveDisc,
                                        saveOptimizers=trainingConf.saveOptimizers,
                                        )

            shouldBeSaved, checkpointName = modelSaveTracker.shouldModelBeSaved(i, epochCounter)
            if (saveModel and shouldBeSaved):
                Utils.saveGannSetup(model, saveName, checkpointName, discriminatorOptim, generatorOptim,
                                    datasetConf=datasetConf,
                                    trainingConf=trainingConf,
                                    saveGen=genDiscUpdateSchedule.saveGen, saveDisc=genDiscUpdateSchedule.saveDisc,
                                    saveOptimizers=trainingConf.saveOptimizers,
                                    )

        epochCounter += 1
        if (saveModel and modelSaveTracker.shouldModelBeSavedAtThisEpoch(epochCounter)):
            Utils.saveGannSetup(model, saveName, "Epoch-{}".format(epochCounter), discriminatorOptim, generatorOptim,
                                datasetConf=datasetConf, trainingConf=trainingConf,
                                saveGen=genDiscUpdateSchedule.saveGen, saveDisc=genDiscUpdateSchedule.saveDisc,
                                saveOptimizers=trainingConf.saveOptimizers,
                                )
