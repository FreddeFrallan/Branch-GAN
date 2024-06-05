import torch


def calculateAdvantageMask(advantages, policyProbs, powerWeight=0.05):
    rewardMask = torch.greater_equal(advantages, 0).float()
    invertedRewardMask = 1 - rewardMask
    rewardProbFactorMask = policyProbs * invertedRewardMask + rewardMask
    return rewardProbFactorMask ** powerWeight


def calculateMinimumProbLimit(advantages, policyProbs, negPushProbLimit=0.1):
    rewardMask = torch.greater_equal(advantages, 0).float()
    invertedRewardMask = 1 - rewardMask

    negativeProbs = policyProbs * invertedRewardMask
    negRewardMask = torch.where(negativeProbs >= negPushProbLimit, invertedRewardMask, rewardMask)

    return rewardMask + negRewardMask


def _mirroredReinforceLoss(policyProbs, advantages, eps=1e-6):
    policyProbs = torch.clamp(policyProbs, eps, 1 - eps)
    logProbs = torch.log(-policyProbs + 1)
    tempLoss = -logProbs * advantages
    return tempLoss


def mirroredNegativeReinforceLoss(policyProbs, advantages, maskZoneC):
    # calculating the normal reinforce loss
    normalLoss = reinforceLoss(policyProbs, advantages, useNegativeEntropyMask=False, useMinProbLimit=False)
    # calculating the mirrored reinforce loss
    mirroredLoss = _mirroredReinforceLoss(policyProbs, advantages)

    # calculate the inverse of maskZoneC
    inverseMaskZoneC = 1 - maskZoneC

    # applying the masks to the losses
    maskedNormalLoss = normalLoss * inverseMaskZoneC
    maskedMirroredLoss = mirroredLoss * maskZoneC

    combinedLoss = maskedNormalLoss + maskedMirroredLoss
    return combinedLoss


def reinforceLoss(policyProbs, advantages, useNegativeEntropyMask, useMinProbLimit, negPushProbLimit=1e-8, eps=1e-6):
    policyProbs = torch.clamp(policyProbs, eps, 1 - eps)
    logProbs = torch.log(policyProbs)
    tempLoss = -logProbs * advantages
    if (useMinProbLimit):
        tempLoss *= calculateMinimumProbLimit(advantages, policyProbs, negPushProbLimit)

    return tempLoss
