import torch


def calcDicValueHeadMetrics(forwardPassData):
    if ('DiscValueHeads' not in forwardPassData):
        return {}

    data = forwardPassData['DiscValueHeads']
    txtInputPreds, altInputPreds = data['txtInputPreds'], data['altInputPreds']
    txtAlterationsPreds, altAlterationsPreds = data['txtAlterationsPreds'], data['altAlterationsPreds']
    return {'Txt-InputValueHead-Mean': txtInputPreds.mean(), 'Alt-InputValueHead-Mean': altInputPreds.mean(),
            'Txt-AltValueHead-Mean': txtAlterationsPreds.mean(), 'Alt-AltValueHead-Mean': altAlterationsPreds.mean(),
            }


def _discInputPlacementMetric(discInputPreds, discAlterationPreds):
    allPreds = torch.cat((discInputPreds.unsqueeze(1), discAlterationPreds), dim=1)
    sortedPreds = torch.argsort(allPreds, dim=1, descending=True).float()
    orderIDs = torch.reshape(torch.arange(allPreds.shape[1]), (1, allPreds.shape[1], 1))
    orderIDs = torch.repeat_interleave(orderIDs, discAlterationPreds.shape[0], dim=0)
    orderIDs = torch.repeat_interleave(orderIDs, discAlterationPreds.shape[-1], dim=-1).float()
    inputPredPlacement = torch.where(sortedPreds.cpu() == 0, orderIDs, torch.zeros(sortedPreds.shape))
    inputPredPlacement = torch.sum(inputPredPlacement, dim=1)

    inputFirstPlacements = torch.where(inputPredPlacement == 0, torch.ones(inputPredPlacement.shape),
                                       torch.zeros(inputPredPlacement.shape))

    inputFirstPlacementsRate = torch.mean(inputFirstPlacements)

    return inputPredPlacement, inputFirstPlacementsRate


def calcAdvantageRatios(advantages):
    inputAdvantages = advantages[:, :1]
    alterationAdvantages = advantages[:, 1:]

    device = advantages.device
    inputAdvantagesMask = torch.where(inputAdvantages > 0,
                                      torch.ones(inputAdvantages.shape).to(device),
                                      torch.zeros(inputAdvantages.shape).to(device),
                                      )
    alterationAdvantagesMask = torch.where(alterationAdvantages > 0,
                                           torch.ones(alterationAdvantages.shape).to(device),
                                           torch.zeros(alterationAdvantages.shape).to(device),
                                           )

    return torch.mean(inputAdvantagesMask), torch.mean(alterationAdvantagesMask)


def calcDiscriminatorMetrics(discInputPreds, discAlterationPreds, forwardPassData):
    discInputPreds = discInputPreds[:, 1:]
    discAlterationPreds = discAlterationPreds[:, :, 1:]

    inputAnswers = torch.greater_equal(discInputPreds, 0.5)
    alterationAnswers = torch.greater_equal(discAlterationPreds, 0.5)

    inputAcc = torch.mean(inputAnswers.float())
    alterationAcc = 1 - torch.mean(alterationAnswers.float())
    firstSamplalterationAcc = 1 - torch.mean(alterationAnswers[:, 0].float())

    inputPredPlacement, inputFirstPlacementsRate = _discInputPlacementMetric(discInputPreds, discAlterationPreds)

    results = {'Input-Acc': inputAcc, 'Alteration-Acc': alterationAcc,
               'Alteration-FirstSample-Acc': firstSamplalterationAcc,
               'Input-Mean': discInputPreds.mean(),
               'Input-STD': discInputPreds.std(),
               'Alteration-Mean': discAlterationPreds.mean(),
               'Alteration-STD': discAlterationPreds.std(),
               'Input-PredPlacement-Mean': inputPredPlacement.mean(),
               'Input-PredPlacement-STD': inputPredPlacement.std(),
               'Input-PredPlacement-FirstRate': inputFirstPlacementsRate,
               }

    for k, v in calcDicValueHeadMetrics(forwardPassData).items():
        results[k] = v

    return results
