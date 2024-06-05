import torch
import os


def _loadStateDictFromPotentialDistributedModel(path):
    stateDisct = torch.load(path, map_location=torch.device('cpu'))
    return {k.replace('module.', ''): v for k, v in stateDisct.items()}


def initModel(generatorConf, discriminatorConf, trainingConf):
    from Models import GanSetup

    model = GanSetup.GannSetup(generatorConf, discriminatorConf,
                               device=trainingConf.device,
                               generatorDevice=trainingConf.kwargs.get('GeneratorDevice', None),
                               discriminatorDevice=trainingConf.kwargs.get('DiscriminatorDevice', None),
                               mleLossWeight=trainingConf.mleLossWeight,
                               rlLossWeight=trainingConf.rlLossWeight,
                               discPredLossWeight=trainingConf.discPredLossWeight,
                               discValueHeadLossWeight=trainingConf.discValueHeadLossWeight,
                               samplingConf=trainingConf.trainingSamplingConf,
                               generatorTeacherForcingInput=trainingConf.generatorTeacherForcing,
                               )

    if (trainingConf.genStartWeights is not None):
        print("Using generator start weights:", trainingConf.genStartWeights)
        model.generator.load_state_dict(_loadStateDictFromPotentialDistributedModel(trainingConf.genStartWeights))
    if (trainingConf.discStartWeights is not None):
        print("Using discriminator start weights:", trainingConf.discStartWeights)
        model.discriminator.load_state_dict(
            _loadStateDictFromPotentialDistributedModel(trainingConf.discStartWeights)
        )
    if (trainingConf.setupStartWeights is not None):
        model.load_state_dict(torch.load(trainingConf.setupStartWeights, map_location=torch.device('cpu')))

    return model


def saveGannSetup(model, runName, timeMarker, discOptimizer=None, genOptimizer=None, mainSavePath='',
                  datasetConf=None, tokenizer=None, saveDisc=True, saveGen=True, saveOptimizers=True):
    print("Saving model:", runName)
    folderName = runName.replace('/', '-')
    mainSaveFolder = os.path.join(mainSavePath, folderName)
    if (os.path.exists(mainSaveFolder) == False):
        os.makedirs(mainSaveFolder)

    # Save configs if they're yet to be created
    discConfPath = os.path.join(mainSaveFolder, 'Discriminator-Config.json')
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
    if (saveOptimizers):
        for optimizer, saveFolder, doSave in [(discOptimizer, discriminatorFolder, saveDisc),
                                              (genOptimizer, generatorFolder, saveGen)]:
            if (doSave and optimizer is not None):
                optimizer.saveOptimizer(os.path.join(saveFolder, 'Optimizer.pt'))
