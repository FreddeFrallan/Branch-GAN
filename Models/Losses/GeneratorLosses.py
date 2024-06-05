from Configs.ReinforceLabelMode import ReinforcementLabelMode
from Models.Losses import AdvantageCalculations, ReinforceLoss
from transformers.models.gpt2 import modeling_gpt2
import torch


def _calculateAdvantages(discInputScores, discAlterationScores, forwardPassData, advantageSectionWeights,
                         inputIncorrectValueHeadsPolicy, alterationIncorrectValueHeadsPolicy, teacherForcingMode=True,
                         ):
    data = forwardPassData['DiscValueHeads']

    # We detach all Discriminator tensors, since there should be no gradient for the discriminator here.
    f = lambda x: x.detach().to(discInputScores.device)
    if (teacherForcingMode):
        discAlterationScores = f(discAlterationScores)
        discInputScores = torch.ones(discInputScores.shape, dtype=discInputScores.dtype).to(discInputScores.device)
    else:
        discInputScores, discAlterationScores = f(discInputScores), f(discAlterationScores)

    txtInputPreds, altInputPreds = f(data['txtInputPreds']), f(data['altInputPreds'])
    txtAlterationsPreds, altAlterationsPreds = f(data['txtAlterationsPreds']), f(data['altAlterationsPreds'])


    advantagesData = AdvantageCalculations.dualHeadAdvantages(discInputScores, discAlterationScores, txtInputPreds,
                                                              txtAlterationsPreds,
                                                              advantageSectionWeights,
                                                              inputIncorrectValueHeadsPolicy,
                                                              alterationIncorrectValueHeadsPolicy)


    return advantagesData


def _calculateDebuggingStats(rlLossesInput, rlLossesAlterations, inAdvantages, altAdvantages, zoneAMask, zoneBMask,
                             zoneCMask, advantageCorrectMask, inputCorrectMask, depthMask=None):
    f = lambda x: torch.sum((rlLossesInput * x[:, :, :-1])) / x[:, :, :-1].sum() if x[:, :, :-1].sum() > 0 else 0
    zoneALossInput, zoneBLossInput, zoneCLossInput = f(zoneAMask[:, :1]), f(zoneBMask[:, :1]), f(zoneCMask[:, :1])
    f = lambda x: torch.sum((rlLossesAlterations * x)) / x.sum() if x.sum() > 0 else 0
    zoneALossAlt, zoneBLossAlt, zoneCLossAlt = f(zoneAMask[:, 1:]), f(zoneBMask[:, 1:]), f(zoneCMask[:, 1:])

    # Debugging Advantages
    f = lambda x: torch.sum((inAdvantages * x)) / x.sum() if x.sum() > 0 else 0
    zoneAAdvantageInput, zoneBLAdvantageInput = f(zoneAMask[:, :1]), f(zoneBMask[:, :1])
    zoneCLAdvantageInput = f(zoneCMask[:, :1])
    f = lambda x: torch.sum((altAdvantages * x)) / x.sum() if x.sum() > 0 else 0
    zoneAAdvantageAlt, zoneBLAdvantageAlt = f(zoneAMask[:, 1:]), f(zoneBMask[:, 1:])
    zoneCAdvantageAlt = f(zoneCMask[:, 1:])

    def zoneMaskCalculation(zone):
        if (depthMask is None):
            return zone.mean()

        if (len(zone.shape) == 3):
            overlap = zone * depthMask.unsqueeze(0)
            depthExtraFactor = zone.shape[0]
        else:
            overlap = zone * depthMask
            depthExtraFactor = 1

        divideBy = depthMask.sum() * depthExtraFactor
        if (divideBy == 0):
            return 0
        return overlap.sum() / divideBy

    # print(zoneAMask.shape, zoneBMask.shape, zoneCMask.shape, advantageCorrectMask.shape, inputCorrectMask.shape)
    return {
        'Rl-Loss-A-Input': zoneALossInput, 'Rl-Loss-B-Input': zoneBLossInput, 'RL-Loss-C-Input': zoneCLossInput,
        'Rl-Loss-A-Alt': zoneALossAlt, 'Rl-Loss-B-Alt': zoneBLossAlt, 'RL-Loss-C-Alt': zoneCLossAlt,
        'Rl-Advantage-A-Input': zoneAAdvantageInput, 'Rl-Advantage-B-Input': zoneBLAdvantageInput,
        'RL-Advantage-C-Input': zoneCLAdvantageInput,
        'Rl-Advantage-A-Alt': zoneAAdvantageAlt, 'Rl-Advantage-B-Alt': zoneBLAdvantageAlt,
        'RL-Advantage-C-Alt': zoneCAdvantageAlt,
        'RL-ZoneA-Input-Factor': zoneAMask[:, :1].mean(),
        'RL-ZoneB-Input-Factor': zoneBMask[:, :1].mean(),
        'RL-ZoneC-Input-Factor': zoneCMask[:, :1].mean(),
        'RL-ZoneA-Alt-Factor': zoneMaskCalculation(zoneAMask[:, 1:]),
        'RL-ZoneB-Alt-Factor': zoneMaskCalculation(zoneBMask[:, 1:]),
        'RL-ZoneC-Alt-Factor': zoneMaskCalculation(zoneCMask[:, 1:]),
        'RL-CorrectValueHeads-Factor': advantageCorrectMask.mean(),
        'RL-CorrectInput-Factor': inputCorrectMask.mean(),
    }


def _reinforceLoss(inputIDs, generatorProbs, discInputScores, discAlterationScores,
                   forwardPassData, reinforcementLearningConfig, depthMask=None):
    inAdvantages, altAdvantages, infoMasks = _calculateAdvantages(discInputScores, discAlterationScores,
                                                                  forwardPassData,
                                                                  reinforcementLearningConfig.advantageSectionWeights,
                                                                  reinforcementLearningConfig.inputIncorrectValueHeadsPolicy,
                                                                  reinforcementLearningConfig.alterationIncorrectValueHeadsPolicy,
                                                                  teacherForcingMode=reinforcementLearningConfig.teacherForcing,
                                                                  )
    zoneAMask, zoneBMask, zoneCMask, advantageCorrectMask, inputCorrectMask = infoMasks

    # We shift the probabilities by one, so that the probabilities match the input IDs
    inputProbs = torch.gather(generatorProbs[:, :-1], dim=-1, index=inputIDs[:, 1:].unsqueeze(-1))
    inputProbs = torch.permute(inputProbs, (0, 2, 1))

    inputCMask = zoneCMask[:, :1]
    alterationCMask = zoneCMask[:, 1:]
    rlLossesInput = ReinforceLoss.mirroredNegativeReinforceLoss(inputProbs, inAdvantages[:, :, :-1],
                                                                inputCMask[:, :, :-1])


    alterationProbs = forwardPassData['AlterationProbs']


    rlLossesAlterations = ReinforceLoss.mirroredNegativeReinforceLoss(alterationProbs, altAdvantages, alterationCMask)

    debugData = _calculateDebuggingStats(rlLossesInput, rlLossesAlterations, inAdvantages,
                                         altAdvantages, zoneAMask, zoneBMask, zoneCMask,
                                         advantageCorrectMask, inputCorrectMask, depthMask=depthMask)

    if (depthMask is None):
        rlLossesAlterations = rlLossesAlterations[:, :, :-1]
        alterationLoss = torch.mean(rlLossesAlterations)
    else:
        rlLossesAlterations = rlLossesAlterations * depthMask
        alterationLoss = torch.sum(rlLossesAlterations) / depthMask.sum()

    inputLoss = rlLossesInput.mean()
    combined = (inputLoss + alterationLoss) / 2
    unweightedCombined = (inputLoss + alterationLoss) / (1 + rlLossesAlterations.shape[1])
    losses = {
        'input': inputLoss,
        'alterations': alterationLoss,
        'combined': combined,
        'unweightedCombined': unweightedCombined
    }

    return losses, altAdvantages, debugData

def _mleLossFromLogits(inputIDs, logits):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputIDs[..., 1:].contiguous()
    loss_fct = modeling_gpt2.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def _calcGeneratorLossNoRL(inputIDs, generatorLogits):
    mleLoss = _mleLossFromLogits(inputIDs, generatorLogits)

    return mleLoss


def calcGeneratorLoss(gannSetup, inputIDs, generatorLogits, generatorProbs,
                      discInputScores, discAlterationScores, forwardPassData, reinforcementLearningConfig,
                      depthMask=None):
    generatorMLELoss = _calcGeneratorLossNoRL(inputIDs, generatorLogits)
    genLosses = {'loss-MLE': generatorMLELoss}

    if (gannSetup.generatorRlLossWeight == 0):
        generatorLoss = generatorMLELoss * gannSetup.generatorMleLossWeight
    else:
        rlLosses, advantages, rlLossInfo = _reinforceLoss(inputIDs, generatorProbs,
                                                          discInputScores, discAlterationScores,
                                                          forwardPassData,
                                                          reinforcementLearningConfig,
                                                          depthMask=depthMask)

        for k, v in rlLossInfo.items():
            genLosses[k] = v

        for k, v in rlLosses.items():
            genLosses['loss-RL-{}'.format(k)] = v

        if (reinforcementLearningConfig.labelMode == ReinforcementLabelMode.ALTERATIONS_ONLY):
            rlLoss = rlLosses['alterations']
        elif (reinforcementLearningConfig.labelMode == ReinforcementLabelMode.INPUT_ONLY):
            rlLoss = rlLosses['input']
        elif (reinforcementLearningConfig.labelMode == ReinforcementLabelMode.ALTERATIONS_AND_INPUT):
            rlLoss = rlLosses['combined']
        elif (reinforcementLearningConfig.labelMode == ReinforcementLabelMode.ALTERATIONS_AND_INPUT_UNWEIGHTED):
            rlLoss = rlLosses['unweightedCombined']

        genLosses['loss-RL'] = rlLoss
        generatorLoss = generatorMLELoss * gannSetup.generatorMleLossWeight + rlLoss * gannSetup.generatorRlLossWeight

    genLosses['loss'] = generatorLoss
    return genLosses
