import torch


def _calcDiscriminatorValueHeadLoss(gannSetup, inputScores, alterationScores, dicAlterationLossWeights,
                                    forwardPassData):
    data = forwardPassData['DiscValueHeads']
    txtInputPreds, altInputPreds = data['txtInputPreds'], data['altInputPreds']
    txtAlterationsPreds, altAlterationsPreds = data['txtAlterationsPreds'], data['altAlterationsPreds']

    # We assume that the input scores already have the first time step removed
    inputLabels = inputScores.detach()
    inputLoss = gannSetup.discValueHeadLoss(txtInputPreds[:, :-1].squeeze(-1), inputLabels)

    alterationScores = alterationScores.detach()
    rootLinks, alterationLinks = forwardPassData['SubNodeStructure']

    # Solve root prediction independently, where we skip the final prediction step, since it's out of bounds
    targetPreds = torch.stack([alterationScores[:, i] for i in rootLinks], dim=0).mean(dim=0)

    # Cast target preds to the data type of txtAlterationsPreds
    targetPreds = targetPreds.to(txtAlterationsPreds.dtype)
    rootLoss = gannSetup.discValueHeadLoss(txtAlterationsPreds[:, :-1].squeeze(-1), targetPreds[:, 1:])

    # For each non-root node we collect its child nodes as targets
    nodeLosses = []
    f = lambda x: torch.stack([alterationScores[:, i] for i in x], dim=0).mean(dim=0).squeeze(-1)
    for i, ids in enumerate(alterationLinks):
        if (len(ids) > 0):
            nodeLosses.append(gannSetup.discValueHeadLoss(altAlterationsPreds[i].squeeze(-1), f(ids),
                                                          weight=dicAlterationLossWeights[i].unsqueeze(0))
                              )
    altLoss = torch.stack(nodeLosses + [rootLoss]).mean()

    return (inputLoss + altLoss) / 2, {'ValueHead-InputLoss': inputLoss, 'ValueHead-AltLoss': altLoss}


def calcDiscriminatorLoss(gannSetup, inputScores, alterationScores, numNegativeUpdateSamples, discDepthMask,
                          discDepthWeights, collectTreeDepthData=False, forwardPassData=None):
    additionalInfo = {}

    if (discDepthMask is not None):
        discDepthMask = discDepthMask.to(inputScores.device)

    # Ignore the first value for input, since it's always correct text
    inputScores = inputScores[:, 1:]
    dicAlterationLossWeights = discDepthMask
    targetDiscAlterationWeights = dicAlterationLossWeights[:numNegativeUpdateSamples]

    inputLabels = torch.ones(inputScores.shape, dtype=inputScores.dtype).to(inputScores.device)
    inputLoss = gannSetup.discriminatorLoss(inputScores, inputLabels)

    if (discDepthWeights is not None):
        alterationLabels = torch.repeat_interleave(discDepthWeights.unsqueeze(0), alterationScores.shape[0], dim=0)
    else:
        alterationLabels = torch.zeros(alterationScores.shape).to(inputScores.device)

    # print("Pre:", alterationScores.shape, alterationLabels.shape, dicAlterationLossWeights.shape)
    # This currently does not play that well with the multi-step approach
    targetScores = alterationScores[:, :numNegativeUpdateSamples]
    targetLabels = alterationLabels[:, :numNegativeUpdateSamples]
    # print("Post:", targetScores.shape, targetLabels.shape, targetDiscAlterationWeights.shape)

    alterationLoss = gannSetup.discriminatorLoss(targetScores, targetLabels, targetDiscAlterationWeights)
    additionalInfo['numAlterationSamples'] = alterationScores[:, :numNegativeUpdateSamples].shape[1]

    discValueHeadLoss, info = _calcDiscriminatorValueHeadLoss(gannSetup, inputScores, alterationScores,
                                                              dicAlterationLossWeights,
                                                              forwardPassData)
    additionalInfo['ValueHeadLoss'] = discValueHeadLoss
    for k, v in info.items():
        additionalInfo[k] = v

    if (collectTreeDepthData == False):
        return inputLoss, alterationLoss, additionalInfo

    with torch.no_grad():
        weightedAlterationLoss = gannSetup.discriminatorLoss(alterationScores,
                                                             alterationLabels, dicAlterationLossWeights,
                                                             reduction='none')

        batchSize = alterationScores.shape[0]
        alterationAcc = alterationScores[:, :numNegativeUpdateSamples] < 0.5
        alterationAcc = torch.sum(alterationAcc * discDepthMask, dim=(0, 2))
        alterationAcc = alterationAcc.float() / (torch.sum(discDepthMask, dim=-1) * batchSize)

        alterationPreds = torch.sum(alterationScores[:, :numNegativeUpdateSamples], dim=(0, 2))
        alterationPreds = alterationPreds / (torch.sum(discDepthMask, dim=-1) * batchSize)

        # Collect losses for each depth, with a separate category for the random nodes
        treeNodeInfo = forwardPassData['TreeNodeInfo']
        depth2ids, depth2randIDs, depth2acc, depth2Pred = {}, {}, {}, {}
        for i, (depth, rand) in enumerate(treeNodeInfo):
            target = depth2randIDs if rand else depth2ids
            if (depth not in target):
                target[depth] = []
            target[depth].append(i)
            if (depth not in depth2acc):
                depth2acc[depth] = []
            depth2acc[depth].append(alterationAcc[i])
            if (depth not in depth2Pred):
                depth2Pred[depth] = []
            depth2Pred[depth].append(alterationPreds[i])

        f = lambda x: torch.stack(x).mean()
        for k, v in depth2ids.items():
            additionalInfo['Depth-{}-Loss'.format(k)] = f([weightedAlterationLoss[:, i] for i in v])
        for k, v in depth2randIDs.items():
            additionalInfo['Depth-{}-RandLoss'.format(k)] = f([weightedAlterationLoss[:, i] for i in v])
        for k, v in depth2acc.items():
            additionalInfo['Depth-{}-Acc'.format(k)] = f(v)
        for k, v in depth2Pred.items():
            additionalInfo['Depth-{}-Pred'.format(k)] = f(v)

    return inputLoss, alterationLoss, additionalInfo
