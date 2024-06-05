from ConfigInstances import PileTraining
from Training import SpawnedExperiment


if __name__ == '__main__':
    # from Models import WeightsUtils
    # WeightsUtils.createNonResModelFromHFModel(
    #     'EleutherAI/pythia-14m',
    #     'GannSetup-Pythia-14m',
    # )


    trainingConf = PileTraining.trainingConf
    # trainingConf.device = ['cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
    trainingConf.device = 'cuda:4'
    trainingConf.doLogData = False

    SpawnedExperiment.main(
        datasetConf=PileTraining.datasetConf,
        generatorConf=PileTraining.generatorConf,
        discriminatorConf=PileTraining.discriminatorConf,
        trainingConf=trainingConf,
    )