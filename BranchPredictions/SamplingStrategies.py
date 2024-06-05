from BranchPredictions import SampleBuffer
from enum import Enum
import torch

SAMPLE_BUFFER = None


def getRand(size, device):
    global SAMPLE_BUFFER
    if (SAMPLE_BUFFER is None):
        SAMPLE_BUFFER = SampleBuffer.SamplingBuffer(device)

    return SAMPLE_BUFFER.rand(size)


DEBUG_TIMING = {
    'SamplingBlockLabels': [],
    'SamplingGenerateIDs': [],
    'SamplingStack': [],
    'sampleNoise': [],
    'sampledProbs': [],
    'argSort': [],
    'sampleNoise2GPU': [],
}

SHAPE_RAND_BUFFER = {}


class SamplingTechniques(Enum):
    RANDOM = 0
    SAMPLE = 1
    TOP_K = 2
    SAMPLE_NO_REPLACEMENT = 3

    EPSILON_GREEDY = 4
    EPSILON_SAMPLE = 5
    EPSILON_SAMPLE_NO_REPLACEMENT = 6

    DEBUG_GRAB_FIRST_IDS = 7

    INVERSE_TOP_K_SAMPLE_NO_REPLACEMENT = 8
    TOP_P_SAMPLE_NO_REPLACEMENT = 9


RANDOM_TECHNIQUES = {
    SamplingTechniques.RANDOM.name: 1,
    SamplingTechniques.EPSILON_GREEDY.name: 1,
    SamplingTechniques.EPSILON_SAMPLE.name: 1,
    SamplingTechniques.EPSILON_SAMPLE_NO_REPLACEMENT.name: 1,
}


class SamplingRule:

    def __init__(self, numAlterations, samplingStrategy, samplingParams, continueFromRandom=False,
                 firstBeamSamplingStrategy=None, firstBeamSamplingParams=None):
        self.numAlterations = numAlterations
        self.samplingStrategy = samplingStrategy
        self.samplingParams = samplingParams
        self.continueFromRandom = continueFromRandom

        self.firstBeamSamplingStrategy = firstBeamSamplingStrategy
        self.firstBeamSamplingParams = {} if firstBeamSamplingParams is None else firstBeamSamplingParams

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return {'NumAlterations': self.numAlterations, 'Strategy': self.samplingStrategy.name,
                'Params': self.samplingParams, 'ContinueFromRandom': self.continueFromRandom,
                'FirstBeam SamplingParams': self.firstBeamSamplingParams,
                'FirstBeam SamplingStrategy': 'None' if self.firstBeamSamplingStrategy is None
                else self.firstBeamSamplingStrategy.name
                }


def apply_temperature(tensor, temperature):
    """
    Apply temperature to a probability tensor.

    Args:
    tensor (torch.Tensor): A probability tensor.
    temperature (float): Temperature to be applied.

    Returns:
    torch.Tensor: A new tensor with temperature applied.
    """
    if temperature == 0:
        raise ValueError("Temperature must be non-zero.")

    # If temperature is 1, return the original tensor
    if temperature == 1:
        return tensor

    # Apply the temperature
    temp_adjusted_tensor = torch.exp(torch.log(tensor) / temperature)

    # Renormalize the tensor to maintain a valid probability distribution along the last dimension
    return temp_adjusted_tensor / torch.sum(temp_adjusted_tensor, dim=-1, keepdim=True)


def apply_noise(tensor, noise_level):
    """
    Apply noise to a probability tensor.

    Args:
    tensor (torch.Tensor): A probability tensor.
    noise_level (float): Percentage of noise to be applied, should be between 0 and 1.

    Returns:
    torch.Tensor: A new tensor with noise applied.
    """
    if noise_level < 0 or noise_level > 1:
        raise ValueError("Noise level must be between 0 and 1.")

    # If noise level is 0, return the original tensor
    if noise_level == 0:
        return tensor

    # Create a tensor of the same size filled with the noise_level
    alpha = torch.ones_like(tensor) * noise_level

    # Sample from a Dirichlet distribution
    dirichlet = torch.distributions.dirichlet.Dirichlet(alpha)
    noise = dirichlet.sample()
    # Blend the original tensor and the noise tensor
    noisy_tensor = tensor * (1 - noise_level) + noise * noise_level

    # Renormalize the tensor to maintain a valid probability distribution along the last dimension
    return noisy_tensor / torch.sum(noisy_tensor, dim=-1, keepdim=True)


def _randomSample(inputIDs, maskedProbs, numSamples, blockSamplingOfLabels=True, includeFinalTimeStep=False):
    vocabSize = maskedProbs.shape[-1]
    randSamples = torch.randint(0, vocabSize, (maskedProbs.shape[0], numSamples, maskedProbs.shape[1]))
    randSamples = randSamples.to(inputIDs.device)
    # TODO: Implement this
    # if (blockSamplingOfLabels):
    #     for i in range(numSamples):
    #         if (includeFinalTimeStep):
    #             overlap = torch.eq(inputIDs, randSamples[:, i])
    #         else:
    #             overlap = torch.eq(inputIDs[:, 1:], randSamples[:, i])
    #         randSamples[:, i] = (randSamples[:, i] + overlap) % vocabSize

    return randSamples


def _sampleWithoutReplacement(maskedProbs, numSamples):
    # Using the exponential sort trick
    # Explained here: https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/

    sampleNoise = getRand(maskedProbs.shape, maskedProbs.device)
    sampleNoise = sampleNoise.to(maskedProbs.device)
    sampledProbs = maskedProbs / sampleNoise
    samples = torch.argsort(sampledProbs, dim=-1, descending=True)[:, :, :numSamples]

    return torch.permute(samples, (0, 2, 1))  # New Shape -> (batch, numSamples, seqLen)


def top_p_sampling(probs, numSamples, p=0.9):
    # Sort the probabilities
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create a mask for probabilities that are part of the top-p
    mask = cumulative_probs <= p

    # Create a mask for probabilities that should be zeroed out
    mask[:, :, :-1] = mask[:, :, 1:].clone()  # Shift the mask to the right
    mask[:, :, -1] = 0  # Exclude the last probability

    # Apply the mask to the original indices and get the original order
    mask = torch.gather(mask, dim=-1, index=sorted_indices.argsort())

    # print(probs.shape, mask.shape)

    # Apply the mask to the original probabilities
    maskedProbs = probs * mask.float()
    normalizedProbs = maskedProbs / torch.sum(maskedProbs, dim=-1, keepdim=True)

    # Use the sampleWithoutReplacement function
    samples = _sampleWithoutReplacement(normalizedProbs, numSamples)

    return samples


def _sampleDistribution(maskedProbs, numSamples):
    distribution = torch.distributions.Categorical(maskedProbs)
    samples = distribution.sample((numSamples,))  # Samples Shape -> (nSamples, batch, seq-1)
    samples = torch.permute(samples, (1, 0, 2))  # Return Shape -> (batch, nSamples, seq-1)
    return samples


def _topKSampling(maskedProbs, numSamples):
    # t0 = time.time()
    sortedArgs = torch.argsort(maskedProbs, dim=-1, descending=True)
    # t1 = time.time()
    samples = sortedArgs[:, :, :numSamples]
    # t2 = time.time()
    # DEBUG_TIMING['argSort'].append(t1 - t0)
    # DEBUG_TIMING['sampleNoise'].append(t2 - t1)
    return torch.permute(samples, (0, 2, 1))  # New Shape -> (batch, numSamples, seqLen)


def _frequencyBalancedTopK(maskedProbs, numSamples, frequenceyCounter):
    # Mask Probs -> (numSamples,seqLen, vocabSize)
    freqDisctribution = frequenceyCounter / frequenceyCounter.sum()
    freqDisctribution = torch.pow(freqDisctribution, 10)
    freqDisctribution = freqDisctribution / freqDisctribution.sum()
    balancedProbs = maskedProbs - freqDisctribution.unsqueeze(0).unsqueeze(0).to(maskedProbs.device) * 5
    return _topKSampling(balancedProbs, numSamples)


def _epsilonGreedy(inputIDs, maskedProbs, numSamples, numRandomSamples=4):
    greedySamples = _topKSampling(maskedProbs, numSamples - numRandomSamples)
    randomSamples = _randomSample(inputIDs, maskedProbs, numRandomSamples)
    return torch.cat((greedySamples, randomSamples), dim=1)


def _epsilonGreedySample(inputIDs, maskedProbs, numSamples, numRandomSamples=4, **kwargs):
    greedySamples = _sampleDistribution(maskedProbs, numSamples - numRandomSamples)
    randomSamples = _randomSample(inputIDs, maskedProbs, numRandomSamples)
    return torch.cat((greedySamples, randomSamples), dim=1)


def _epsilonGreedySampleNoReplacement(inputIDs, maskedProbs, numSamples, numRandomSamples=4, includeFinalTimeStep=False,
                                      **kwargs):
    distributionSamples = _sampleWithoutReplacement(maskedProbs, numSamples - numRandomSamples)
    randomSamples = _randomSample(inputIDs, maskedProbs, numRandomSamples, includeFinalTimeStep)
    return torch.cat((distributionSamples, randomSamples), dim=1)


def _inverseTopKsampleNoReplacement(maskedProbs, numSamples, topK=3):
    sortedArgs = torch.argsort(maskedProbs, dim=-1, descending=True)
    topKSamples = sortedArgs[:, :, :topK]
    topMaskedProbs = torch.scatter(maskedProbs, dim=-1, index=topKSamples, value=0)
    return _sampleWithoutReplacement(topMaskedProbs, numSamples)


def sampleWithSamplingRule(probs, inputIDs, samplingRule, blockSamplingOfLabels=True, includeFinalTimeStep=False):
    return sampleFromGeneratorDistribution(probs, samplingRule.numAlterations, inputIDs, samplingRule.samplingStrategy,
                                           blockSamplingOfLabels, includeFinalTimeStep, **samplingRule.samplingParams)


def sampleFromGeneratorDistribution(probs, numSamples, inputIDs, samplingTechnique, blockSamplingOfLabels=True,
                                    includeFinalTimeStep=False, **samplingTechniqueParams):
    if ('temperature' in samplingTechniqueParams):
        probs = apply_temperature(probs, samplingTechniqueParams['temperature'])

    if ('noise_level' in samplingTechniqueParams):
        probs = apply_noise(probs, samplingTechniqueParams['noise_level'])

    # Probs Shape -> (batch, seq, vocab)
    # t0 = time.time()
    if (blockSamplingOfLabels):
        if (includeFinalTimeStep):
            maskedProbsPart1 = torch.scatter(probs[:, :-1], dim=-1, index=inputIDs[:, 1:].unsqueeze(-1), value=0)
            maskedProbs = torch.cat((maskedProbsPart1, probs[:, -1:]), dim=1)
        else:
            maskedProbs = torch.scatter(probs[:, :-1], dim=-1, index=inputIDs[:, 1:].unsqueeze(-1), value=0)
    else:
        maskedProbs = probs if includeFinalTimeStep else probs[:, :-1]
    # t1 = time.time()

    if (samplingTechnique == SamplingTechniques.RANDOM):
        samples = _randomSample(inputIDs, maskedProbs, numSamples)
    elif (samplingTechnique == SamplingTechniques.SAMPLE):
        samples = _sampleDistribution(maskedProbs, numSamples)
    elif (samplingTechnique == SamplingTechniques.TOP_K):
        samples = _topKSampling(maskedProbs, numSamples)
    elif (samplingTechnique == SamplingTechniques.EPSILON_GREEDY):
        samples = _epsilonGreedy(inputIDs, maskedProbs, numSamples, **samplingTechniqueParams)
    elif (samplingTechnique == SamplingTechniques.EPSILON_SAMPLE):
        samples = _epsilonGreedySample(inputIDs, maskedProbs, numSamples, **samplingTechniqueParams)
    elif (samplingTechnique == SamplingTechniques.SAMPLE_NO_REPLACEMENT):
        samples = _sampleWithoutReplacement(maskedProbs, numSamples)
    elif (samplingTechnique == SamplingTechniques.EPSILON_SAMPLE_NO_REPLACEMENT):
        samples = _epsilonGreedySampleNoReplacement(inputIDs, maskedProbs, numSamples, **samplingTechniqueParams)
    elif (samplingTechnique == SamplingTechniques.INVERSE_TOP_K_SAMPLE_NO_REPLACEMENT):
        inverseTopK = samplingTechniqueParams.get('inverseTopK', 3)
        samples = _inverseTopKsampleNoReplacement(maskedProbs, numSamples, inverseTopK)
    elif (samplingTechnique == SamplingTechniques.TOP_P_SAMPLE_NO_REPLACEMENT):
        topP = samplingTechniqueParams.get('topP', 0.5)
        samples = top_p_sampling(maskedProbs, numSamples, topP)
    else:
        raise NotImplementedError()

    stackedInputIDs = torch.repeat_interleave(inputIDs[:, :1], numSamples, dim=0)
    stackedInputIDs = torch.reshape(stackedInputIDs, (inputIDs.shape[0], numSamples, 1))
    alterationSequences = torch.cat((stackedInputIDs, samples), dim=-1)
    return alterationSequences, maskedProbs


def sanityCheckSampleWithoutReplacement():
    import numpy as np
    import tqdm
    dummyDistribution = [
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.6, 0.1, 0.1, 0.1, 0.1],
        [0.4, 0, 0.1, 0.1, 0.4],
    ]
    dummyDistribution = torch.tensor(dummyDistribution)
    numSamplesPerTurn = 2

    results = {i: {j: 0 for j in range(dummyDistribution.shape[1])} for i in range(dummyDistribution.shape[0])}
    npResults = {i: {j: 0 for j in range(dummyDistribution.shape[1])} for i in range(dummyDistribution.shape[0])}

    for _ in tqdm.tqdm(range(10000), 'Sanity Checking Sampling'):
        res = _sampleWithoutReplacement(dummyDistribution, numSamplesPerTurn)
        for i, row in enumerate(res):
            for idx in row:
                results[i][idx.item()] += 1

        for i, row in enumerate(dummyDistribution.numpy()):
            res = np.random.choice(range(len(row)), numSamplesPerTurn, replace=False, p=row)
            for idx in res:
                npResults[i][idx] += 1

    for res1, res2 in zip(results.values(), npResults.values()):
        row1 = np.array(list(res1.values()))
        row2 = np.array(list(res2.values()))
        print("------------------------")
        print(row1 / np.sum(row1))
        print(row2 / np.sum(row2))
