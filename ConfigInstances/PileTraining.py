from Configs import TrainingConfig, ReinforceLabelMode
from Training import GanUpdateSchedule, Optimizers
from BranchPredictions import SamplingStrategies
from torch.optim.lr_scheduler import CyclicLR
import transformers


numTestSamples = 256
numEpochs = 100

numTrainSamples = 2000000
updatesPerEvaluation = 2000
numValidationAlterations = 1

# Data for Pythia
vocabSize = 50304
genStartStep = 500

datasetConf = TrainingConfig.DatasetConfig('PileDedup')
additionalValidationSets = []

STANDARD_VALIDATION = {
    'SampleNoRep':
        TrainingConfig.SamplingConfig(
            SamplingStrategies.SamplingRule(
                numAlterations=numValidationAlterations,
                samplingStrategy=SamplingStrategies.SamplingTechniques.SAMPLE_NO_REPLACEMENT,
                samplingParams={},
            )),
    'Random':
        TrainingConfig.SamplingConfig(
            SamplingStrategies.SamplingRule(
                numAlterations=numValidationAlterations,
                samplingStrategy=SamplingStrategies.SamplingTechniques.RANDOM,
                samplingParams={}
            )),
    'Top-K':
        TrainingConfig.SamplingConfig(
            SamplingStrategies.SamplingRule(
                numAlterations=numValidationAlterations,
                samplingStrategy=SamplingStrategies.SamplingTechniques.TOP_K,
                samplingParams={}
            )),
}

trainingConf = TrainingConfig.TrainingConfig(
    runName='BranchGAN-Pile-410m',
    mleLossWeight=1,
    rlLossWeight=1,
    discValueHeadLossWeight=0.25,
    reinforcementLearningConfig=ReinforceLabelMode.ReinforcementLearningConfig(
        labelMode=ReinforceLabelMode.ReinforcementLabelMode.ALTERATIONS_AND_INPUT_UNWEIGHTED,
        sectionWeightA=1, sectionWeightB=1, sectionWeightC=1.5,
        teacherForcing=True,
    ),

    trainingSamplingConf=TrainingConfig.RoundRobinSamplingConfig(
        [
            TrainingConfig.SamplingConfig(
                [
                    SamplingStrategies.SamplingRule(
                        numAlterations=1,
                        samplingStrategy=SamplingStrategies.SamplingTechniques.TOP_K,
                        samplingParams={},
                    )

                    for _ in range(16)
                ],
                includeAllMultiStepSamplesForDisc=True,
                numSampleSequences=32,
            ),

        ]
    ),


    validationSamplingConf=STANDARD_VALIDATION,
    numNegativeUpdateAlterations=9999,

    genOptimConfig=
    TrainingConfig.OptimizerConfig(lossKey='loss', name='adam', lr=0.00001,
                                      useZeroOptimizer=False,
                                      params={'betas': (0.5, 0.999)},
                                      schedulerType=Optimizers.FreezeLRSchedule,
                                      schedulerArgs={'lr_start': 1e-8, 'lr_end': 0.000005,
                                                     'zero_steps': 1000,
                                                     'warmup_steps': 1000
                                                     },
                                      gradAccum=8),
    discOptimConfig=TrainingConfig.OptimizerConfig(lossKey='loss', name='adam', lr=0.0001, gradAccum=8,
                                                      useZeroOptimizer=False,
                                                      schedulerType=CyclicLR,
                                                      schedulerArgs={'base_lr': 1e-8, 'max_lr': 0.00001,
                                                                     'step_size_up': 500,
                                                                     'cycle_momentum': False}
                                                      ),
    negPushProbLimit=9999,
    updatesPerEvaluation=updatesPerEvaluation,

    mixedPrecision=False,
    batchSize=8,
    devicesPerConfig=2,
    validationBatchSize=4,
    # validationBatchSize=8,
    numEpochs=numEpochs,
    genDiscUpdateSchedule=GanUpdateSchedule.AlwaysUpdateSchedule(),
    displayProgression=True, doLogData=True,
    saveModel=False, efficentMode=True,
    saveOptimizers=False,
    additionalModelSaveFreq=None,
    epochSaveFreq=1,

    generationEvaluationConfig=TrainingConfig.TextGenerationConfig(
        maxGenerationLength=96, contextSize=32, maxNumSamples=128,
    ),

    # ************* RL experiments
    # discStartWeights='GannSetup-Pythia-14m/Start/Discriminator/Discriminator.pt',
    # genStartWeights='GannSetup-Pythia-70m/Start/Generator/Generator.pt',
    # discStartWeights='GannSetup-Pythia-70m/Start/Discriminator/Discriminator.pt',
    # genStartWeights='GannSetup-Pythia-410m/Start/Generator/Generator.pt',
    # discStartWeights='GannSetup-Pythia-410m/Start/Discriminator/Discriminator.pt',
    # genStartWeights='GannSetup-Pythia-410m/Start/Generator/Generator.pt',
    # discStartWeights='GannSetup-Pythia-1B/Start/Discriminator/Discriminator.pt',
    # genStartWeights='GannSetup-Pythia-2_8B/Start/Generator/Generator.pt',
)


generatorConf=transformers.AutoConfig.from_pretrained("EleutherAI/pythia-410m-deduped")
discriminatorConf=transformers.AutoConfig.from_pretrained("EleutherAI/pythia-14m")

