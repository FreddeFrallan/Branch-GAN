from Training import TrainingLoop, Optimizers
from DataManagers import DataUtils
from Models import FileManagement
import numpy as np
import torch


def loadValidationDatasets(datasetConf, trainingConf):
    validationBatchSize = trainingConf.batchSize if trainingConf.validationBatchSize is None else trainingConf.validationBatchSize
    testDataloader = DataUtils.loadPreTokenizedDataloader(validationBatchSize, datasetConf.datasetPath,
                                                          maxNumSamples=datasetConf.maxNumSamples,
                                                          )
    return testDataloader


def initOptimizers(discriminator, generator, discOptimizers, genOptimizers, trainingConfig, trainDataLoader):
    discOptimizers = discOptimizers if type(discOptimizers) == list else [discOptimizers]
    genOptimizers = genOptimizers if type(genOptimizers) == list else [genOptimizers]

    discWrapper = Optimizers.createOptimizationWrappers(discriminator, discOptimizers, trainDataLoader,
                                                        trainingConfig.numEpochs,
                                                        mixedPrecision=trainingConfig.mixedPrecision)
    genWrapper = Optimizers.createOptimizationWrappers(generator, genOptimizers, trainDataLoader,
                                                       trainingConfig.numEpochs,
                                                       mixedPrecision=trainingConfig.mixedPrecision)

    if (trainingConfig.discOptimizerStartWeights is not None):
        print("Using discriminator optimizer start weights:", trainingConfig.genOptimizerStartWeights)
        weights = torch.load(trainingConfig.discOptimizerStartWeights, map_location=torch.device('cpu'))
        discWrapper.optimizer.load_state_dict(weights)
    if (trainingConfig.genOptimizerStartWeights is not None):
        print("Using generator optimizer start weights:", trainingConfig.genOptimizerStartWeights)
        weights = torch.load(trainingConfig.genOptimizerStartWeights, map_location=torch.device('cpu'))
        genWrapper.optimizer.load_state_dict(weights)

    return discWrapper, genWrapper


def initWandbLog(datasetConf, generatorConf, discriminatorConf, trainingConf):
    return {
        'Dataset': datasetConf.to_dict(),
        'General-Training': trainingConf.to_dict(),
        'Discriminator': discriminatorConf.to_dict(),
        'Generator': generatorConf.to_dict(),
    }

def _multiDeviceSpawnFunc(rank, datasetConf, generatorConf, discriminatorConf, trainingConf,modelParallelismDevices):
    import torch.distributed as dist
    import os
    worldSize = len(trainingConf.device) if modelParallelismDevices is None else len(modelParallelismDevices)
    print("Initailizing process group", os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
    dist.init_process_group('nccl', rank=rank, world_size=worldSize)

    # Set the worker to the correct device and adapt the batch size
    if (modelParallelismDevices is not None):
        localDevices = modelParallelismDevices[rank]
        trainingConf.device = localDevices[0]
        trainingConf.kwargs['GeneratorDevice'] = localDevices[0]
        trainingConf.kwargs['DiscriminatorDevice'] = localDevices[1]
    else:
        trainingConf.device = trainingConf.device[rank]

    batchSizeMin = int(np.floor(trainingConf.batchSize / worldSize))
    rest = int(trainingConf.batchSize % worldSize)
    workerBatchSizes = [batchSizeMin for _ in range(worldSize)]
    for i in range(rest):  # Distribute the rest over the R first chunks
        workerBatchSizes[i] += 1
    trainingConf.batchSize = workerBatchSizes[rank]
    print("Worker: {} using device: {} with a batch size of: {}".format(rank, trainingConf.device,
                                                                        trainingConf.batchSize))

    # disable loging and model saving if we're not rank=0
    if (rank > 0):
        print("Disabling a lot of shit for rank", rank)
        trainingConf.saveModel = False
        trainingConf.doLogData = False
        trainingConf.displayProgression = False

    # with torch.torch.device(trainingConf.device):
    with torch.cuda.device(trainingConf.device):
        _mainProccess(datasetConf, generatorConf, discriminatorConf, trainingConf,
                      ddpWorldSize=worldSize, ddpRank=rank)


def initModelParallelismTraining(datasetConf, generatorConf, discriminatorConf, trainingConf):
    if (len(trainingConf.device) == 2):
        print("Putting generator and discriminator on different devices")
        trainingConf.kwargs['GeneratorDevice'] = trainingConf.device[0]
        trainingConf.kwargs['DiscriminatorDevice'] = trainingConf.device[1]
    else:
        raise NotImplementedError("Higher order model parallelism is not implemented yet.")

    _mainProccess(datasetConf, generatorConf, discriminatorConf, trainingConf)


def _initDataParallelPort(port):
    import os
    if (os.getenv('MASTER_ADDR') is None):
        print("Setting default master address to:", "localhost")
        os.environ['MASTER_ADDR'] = "localhost"
    if (os.getenv('MASTER_PORT') is None):
        print("Setting default master port to:", port)
        os.environ['MASTER_PORT'] = str(port)


def initDataParallelismTraining(datasetConf, generatorConf, discriminatorConf, trainingConf, port=12345):
    import torch.multiprocessing as mp
    _initDataParallelPort(port)
    args = (datasetConf, generatorConf, discriminatorConf, trainingConf, None)
    mp.spawn(_multiDeviceSpawnFunc, args=args, nprocs=len(trainingConf.device), join=True)


def initDataAndModelParallelismTraining(datasetConf, generatorConf, discriminatorConf, trainingConf, port=12345):
    import torch.multiprocessing as mp
    _initDataParallelPort(port)

    # Divide the devices over the model parallelism devices
    modelParallelismDevices = []
    numParallelWorkers = len(trainingConf.device) // trainingConf.devicesPerConfig
    for i in range(numParallelWorkers):
        start = i * trainingConf.devicesPerConfig
        end = (i + 1) * trainingConf.devicesPerConfig
        modelParallelismDevices.append(trainingConf.device[start:end])

    print("Model parallelism devices:", modelParallelismDevices)
    args = (datasetConf, generatorConf, discriminatorConf, trainingConf,modelParallelismDevices)
    mp.spawn(_multiDeviceSpawnFunc, args=args, nprocs=numParallelWorkers, join=True)


def main(datasetConf, generatorConf, discriminatorConf, trainingConf, port=12345):
    # Route to the correct training function depending on the number of training devices, and the devicesPerConfig
    if (type(trainingConf.device) == list and len(trainingConf.device) > 1):
        if (trainingConf.devicesPerConfig == 1):
            print("Starting Data Parallelism Training")
            initDataParallelismTraining(datasetConf, generatorConf, discriminatorConf, trainingConf, port=port)
        elif (trainingConf.devicesPerConfig > 1 and len(trainingConf.device) == trainingConf.devicesPerConfig):
            print("Starting Model Parallelism Training")
            initModelParallelismTraining(datasetConf, generatorConf, discriminatorConf, trainingConf)
        elif (len(trainingConf.device) % trainingConf.devicesPerConfig == 0):
            print("Starting Data and Model Parallelism Training")
            initDataAndModelParallelismTraining(datasetConf, generatorConf, discriminatorConf, trainingConf,port=port)
        else:
            raise ValueError("The number of devices is not a multiple of the number of devices per config")

    else:
        print("Starting single device training")
        if (trainingConf.devicesPerConfig > 1):
            print("**** WARNING ****",
                  "You are using a single device but have set the number of devices per config to be greater than 1")
        _mainProccess(datasetConf, generatorConf, discriminatorConf, trainingConf)


def setup_device_placement(model, trainingConf, ddpRank=None):
    if ('GeneratorDevice' in trainingConf.kwargs):
        print("Moving {} generator to device {}".format(ddpRank, trainingConf.kwargs['GeneratorDevice']))
        model.generator = model.generator.to(trainingConf.kwargs['GeneratorDevice'])
        print("Moving {} discriminator to device {}".format(ddpRank, trainingConf.kwargs['DiscriminatorDevice']))
        model.discriminator = model.discriminator.to(trainingConf.kwargs['DiscriminatorDevice'])
    else:
        model = model.to(trainingConf.device)
        print("Moving model {} to device {}".format(ddpRank, trainingConf.device))
        print("Disc Device:", model.discriminator.nonResidualModel.device)


def _mainProccess(datasetConf, generatorConf, discriminatorConf, trainingConf, ddpWorldSize=None, ddpRank=None):
    import traceback
    trainDataloader = DataUtils.loadPreTokenizedDataloader(trainingConf.batchSize, datasetConf.datasetPath,
                                                           ddpWorldSize=ddpWorldSize, ddpRank=ddpRank,
                                                           maxNumSamples=datasetConf.maxNumSamples
                                                           )
    testDataLoader = loadValidationDatasets(datasetConf, trainingConf)

    model = FileManagement.initModel(generatorConf, discriminatorConf, trainingConf)
    setup_device_placement(model, trainingConf, ddpRank=ddpRank)

    if (ddpRank is not None):
        from torch.nn.parallel import DistributedDataParallel as DDP
        if ('GeneratorDevice' in trainingConf.kwargs):
            print("Creating DDP models for rank {} at device {} and {}".format(ddpRank,
                                                                               trainingConf.kwargs['GeneratorDevice'],
                                                                               trainingConf.kwargs[
                                                                                   'DiscriminatorDevice'])
                  )
            f = lambda x, dev, y=True: DDP(x, device_ids=[dev], find_unused_parameters=y)
            model.generator = f(model.generator, dev=trainingConf.kwargs['GeneratorDevice'])
            model.discriminator = f(model.discriminator, dev=trainingConf.kwargs['DiscriminatorDevice'])
        else:
            print("Creating DDP models for rank {} at device {}".format(ddpRank, trainingConf.device))
            f = lambda x, y=True: DDP(x, device_ids=[trainingConf.device], find_unused_parameters=y)
            model.generator = f(model.generator)
            model.discriminator = f(model.discriminator)

    discriminatorOptim, generatorOptim = initOptimizers(model.discriminator, model.generator,
                                                        trainingConf.discOptimConfig,
                                                        trainingConf.genOptimConfig, trainingConf,
                                                        trainDataloader)

    TrainingLoop.efficentTrainingLoop(model=model,
                                      trainDataset=trainDataloader, testData=testDataLoader,
                                      numEpochs=trainingConf.numEpochs,
                                      batchSize=trainingConf.batchSize, device=trainingConf.device,
                                      generatorOptim=generatorOptim, discriminatorOptim=discriminatorOptim,
                                      datasetConf=datasetConf, trainingConf=trainingConf,
                                      trainingSamplingConf=trainingConf.trainingSamplingConf,
                                      validationSamplingConf=trainingConf.validationSamplingConf,
                                      numUpdateSamples=trainingConf.numNegativeUpdateAlterations,
                                      updatesPerEvaluation=trainingConf.updatesPerEvaluation,
                                      saveName=trainingConf.runName,
                                      doLogData=trainingConf.doLogData,
                                      displayProgression=trainingConf.displayProgression,
                                      genDiscUpdateSchedule=trainingConf.genDiscUpdateSchedule,
                                      saveModel=trainingConf.saveModel,
                                      saveFreq=trainingConf.additionalModelSaveFreq,
                                      ddpRank=ddpRank,
                                      **trainingConf.kwargs,
                                      )
