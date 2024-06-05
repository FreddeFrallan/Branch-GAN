from Configs.ReinforceLabelMode import IncorrectValueHeadsPolicy
import torch


def dualHeadAdvantages(discInputScores, alterationScores, inputValueHeadPred, alterationValueHeadPred,
                       advantageSectionWeights, inputIncorrectValueHeadsPolicy, alterationIncorrectValueHeadsPolicy,
                       minZoneCLoss=0.1, epsilon=0.00001, min_denominator_value=1e-5):
    f = lambda x: x.detach().squeeze(-1)
    inputValueHeadPred, alterationValueHeadPred = f(inputValueHeadPred), f(alterationValueHeadPred)
    advantageCorrectMask = 1 - torch.greater_equal(alterationValueHeadPred, inputValueHeadPred).int()
    inputCorrectMask = torch.greater_equal(discInputScores, alterationValueHeadPred)

    # We concat the input scores to the alteration scores, but the input scores need to be shifted a step
    # We hence add a dummy step to it
    padDevice, padShape = discInputScores.device, (discInputScores.shape[0], 1)
    paddedInputScores = torch.cat((discInputScores[:, 1:], torch.zeros(padShape).to(padDevice)), dim=1)
    discScores = torch.cat((paddedInputScores.unsqueeze(1), alterationScores), dim=1)

    # Calculate in which zone the predictions fall
    zoneAMask = torch.greater_equal(discScores, inputValueHeadPred.unsqueeze(1))
    greaterThanValuePred = torch.greater_equal(discScores, alterationValueHeadPred.unsqueeze(1))
    zoneCMask = 1 - greaterThanValuePred.int()
    zoneBMask = torch.logical_xor(zoneAMask, greaterThanValuePred)

    # Calculate the values for predictions in zone B (between Input & Alt)
    # The absolute value and epsilon are used to avoid division by zero
    # valueHeadDiff = torch.abs(inputValueHeadPred - alterationValueHeadPred) + epsilon
    valueHeadDiff = torch.clamp(torch.abs(inputValueHeadPred - alterationValueHeadPred) + epsilon,
                                min=min_denominator_value)
    linearScaledZoneB = (discScores - alterationValueHeadPred.unsqueeze(1)) / valueHeadDiff.unsqueeze(1)

    # Combine the zone masks and values to get the advantage scores
    advantageScores = zoneAMask.float() * advantageSectionWeights.A + \
                      linearScaledZoneB * zoneBMask.float() * advantageSectionWeights.B + \
                      zoneCMask.float() * advantageSectionWeights.C

    inputAdvantages, alterationAdvantages = advantageScores[:, :1], advantageScores[:, 1:]

    # If we want to ignore samples where the value heads are incorrect, we mask the advantages
    if (inputIncorrectValueHeadsPolicy == IncorrectValueHeadsPolicy.IGNORE_SAMPLE):
        inputAdvantages = inputAdvantages * advantageCorrectMask.unsqueeze(1)
    if (alterationIncorrectValueHeadsPolicy == IncorrectValueHeadsPolicy.IGNORE_SAMPLE):
        alterationAdvantages = alterationAdvantages * advantageCorrectMask.unsqueeze(1)

    return inputAdvantages, alterationAdvantages, (zoneAMask.float(), zoneBMask.float(), zoneCMask.float(),
                                                   advantageCorrectMask.float(), inputCorrectMask.float())
