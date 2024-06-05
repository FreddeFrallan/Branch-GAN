from Training.Metrics import MetricUtils
import torch


def _calcRankLossMassRatio(inputIDs, generatorProbs, inputPlacements, rankRates=(0, 8, 16, 64, 128, 256, 512, 1024)):
    device = inputIDs.device
    losses = MetricUtils.mleLossFromProbs(inputIDs, generatorProbs, aggrigate=False).squeeze(-1)
    lossSum = losses.sum()
    meanLosses = losses / lossSum

    rankRateResults = {}
    for r in rankRates:
        rankMask = torch.where(inputPlacements <= r,
                               torch.ones(inputPlacements.shape).to(device),
                               torch.zeros(inputPlacements.shape).to(device)
                               )
        maskedLosses = meanLosses * rankMask
        rankRateResults[r] = maskedLosses.sum() / lossSum
    return rankRateResults

    pass


def _calcRankProbMasses(generatorProbs, rankRates=(1, 8, 16, 64, 128, 256, 512, 1024)):
    sortedProbs = torch.sort(generatorProbs, dim=-1, descending=True).values
    return {r: torch.sum(sortedProbs[:, :, :r], dim=-1).mean() for r in rankRates}


def _calculateInputIDsPlacementMetric(inputIDs, generatorProbs, rankRates=(0, 3, 8, 16, 32)):
    sortedProbs = torch.argsort(generatorProbs, dim=-1, descending=True)
    inputIDs, sortedProbs = inputIDs[:, 1:], sortedProbs[:, :-1]
    device = inputIDs.device

    selectedIDs = torch.where(sortedProbs == inputIDs.unsqueeze(-1),
                              torch.ones(sortedProbs.shape).to(device),
                              torch.zeros(sortedProbs.shape).to(device))

    orderIDs = torch.reshape(torch.arange(sortedProbs.shape[-1]), (1, 1, sortedProbs.shape[-1]))
    orderIDs = torch.repeat_interleave(orderIDs, sortedProbs.shape[0], dim=0)
    orderIDs = torch.repeat_interleave(orderIDs, sortedProbs.shape[1], dim=1).float().to(device)
    inputPlacements = torch.sum(orderIDs * selectedIDs, dim=-1)

    rankRateResults = {
        r: torch.where(inputPlacements <= r,
                       torch.ones(inputPlacements.shape).to(device),
                       torch.zeros(inputPlacements.shape).to(device)
                       ).mean()
        for r in rankRates}

    return inputPlacements, rankRateResults


def _calcGeneratorRankAndDiscPredCorrelation(generatorRanks, discInputScores):
    corefStack = torch.stack((generatorRanks.flatten(), discInputScores.flatten()))
    coreff = torch.corrcoef(corefStack)
    return coreff[0, 1]

def calculateGeneratorMetricsNoDiscriminator(inputIDs, generatorProbs, generatorEntropies):
    # # Collect top-k samples, used later for comparing overlap
    inputPlacements, predictionRankRateResults = _calculateInputIDsPlacementMetric(inputIDs, generatorProbs)
    rankProbMasses = _calcRankProbMasses(generatorProbs)
    rankLossRatios = _calcRankLossMassRatio(inputIDs, generatorProbs, inputPlacements)

    labelProbs = torch.gather(generatorProbs[:, :-1], dim=-1, index=inputIDs[:, 1:].unsqueeze(-1))
    minProbs = torch.min(generatorProbs, dim=-1).values
    maxProbs = torch.max(generatorProbs, dim=-1).values
    data = {'Input-Prob-Mean': torch.mean(labelProbs),
            'Input-Prob-STD': torch.std(labelProbs),
            'Entropy-Mean': generatorEntropies.mean(),
            'Entropy-STD': generatorEntropies.std(),

            'Probs-Min-Mean': minProbs.mean(),
            'Probs-Min-Min': minProbs.min(),
            'Probs-Max-Mean': maxProbs.mean(),

            'Input-PredPlacement-Mean': inputPlacements.mean(),
            'Input-PredPlacement-Median': inputPlacements.median(),
            'Input-PredPlacement-STD': inputPlacements.std(),
            }
    for k, v in predictionRankRateResults.items():
        data['Input-PredPlacement-RankRate-{}'.format(k)] = v
    for k, v in rankProbMasses.items():
        data['RankProbMass-{}'.format(k)] = v
    for k, v in rankLossRatios.items():
        data['RankLossRatio-{}'.format(k)] = v
    return data


def calculateGeneratorMetrics(inputIDs, generatorProbs, maskedProbs, discInputScores, discAlterationScores):
    advatageScores = torch.cat((discInputScores[:, 1:].unsqueeze(1), discAlterationScores[:, :, 1:]), dim=1)
    meanAdvantage = torch.mean(advatageScores, dim=1, keepdim=True)

    meanReward = discAlterationScores[:, :, 1:].mean()
    stdReward = discAlterationScores[:, :, 1:].std()
    advantageScores = (discAlterationScores[:, :, 1:] - meanReward)  # / (stdReward + 1e-7)
    meanFirstSampleReward = discAlterationScores[:, 0, 1:].mean()
    stdFirstSampleReward = discAlterationScores[:, 0, 1:].std()

    # # Collect top-k samples, used later for comparing overlap
    sortedProbs = torch.argsort(maskedProbs, dim=-1, descending=True)
    top5 = sortedProbs[:, :, :5]
    top10 = sortedProbs[:, :, :10]
    top20 = sortedProbs[:, :, :20]

    inputPlacements, predictionRankRateResults = _calculateInputIDsPlacementMetric(inputIDs, generatorProbs)
    rankProbMasses = _calcRankProbMasses(generatorProbs)
    rankLossRatios = _calcRankLossMassRatio(inputIDs, generatorProbs, inputPlacements)

    labelProbs = torch.gather(generatorProbs[:, :-1], dim=-1, index=inputIDs[:, 1:].unsqueeze(-1))
    minProbs = torch.min(generatorProbs, dim=-1).values
    maxProbs = torch.max(generatorProbs, dim=-1).values
    data = {'Input-Prob-Mean': torch.mean(labelProbs),
            'Input-Prob-STD': torch.std(labelProbs),
            'RL-MeanReward': meanReward,
            'RL-MeanAdvantage': meanAdvantage,
            'RL-STDReward': stdReward,
            'RL-FirstSample-MeanReward': meanFirstSampleReward,
            'RL-FirstSample-STDReward': stdFirstSampleReward,
            'Advantage-Mean': advantageScores.mean(),

            'Probs-Min-Mean': minProbs.mean(),
            'Probs-Min-Min': minProbs.min(),
            'Probs-Max-Mean': maxProbs.mean(),

            'Input-PredPlacement-Mean': inputPlacements.mean(),
            'Input-PredPlacement-Median': inputPlacements.median(),
            'Input-PredPlacement-STD': inputPlacements.std(),
            # 'Input-PredPlacement-DiscPred-Correlation': genRankDiscPredCorrelation,

            'Top-5': top5,
            'Top-10': top10,
            'Top-20': top20,
            }
    for k, v in predictionRankRateResults.items():
        data['Input-PredPlacement-RankRate-{}'.format(k)] = v
    for k, v in rankProbMasses.items():
        data['RankProbMass-{}'.format(k)] = v
    for k, v in rankLossRatios.items():
        data['RankLossRatio-{}'.format(k)] = v
    return data
