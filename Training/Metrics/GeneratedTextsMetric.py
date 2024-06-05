import numpy as np

def TTR(A, B):
    # Convert the numpy arrays to Python sets for efficient operations
    setA = set(A)
    setB = set(B)

    # Compute the number and fraction of unique numbers in B
    unique_B = len(setB)
    ttrB = unique_B / len(B)

    # Compute the number and fraction of numbers that are unique in both A and B
    unique_AB = len(setB - setA)
    contextIncludedTtrB = unique_AB / len(B)

    # Return the computed values
    return ttrB, contextIncludedTtrB

def calculateTTR(sequencePairs):
    results = []

    # Iterate over the sequence pairs and analyze each pair
    for context, generation in sequencePairs:
        results.append(TTR(context, generation))

    # Convert results to numpy array for easy calculation of statistics
    results = np.array(results)

    meanResults = np.mean(results, axis=0)
    meanTtrB = meanResults[0]
    meanContextIncludedTtrB = meanResults[1]

    return meanTtrB, meanContextIncludedTtrB


def selfCrossEntropy(originalCrossEntropy, generatedCrossEntropy):
    # Compute the mean cross-entropy for the original and generated sequences
    meanOriginalCrossEntropy = np.mean(originalCrossEntropy)
    meanGeneratedCrossEntropy = np.mean(generatedCrossEntropy)
    stdOriginalCrossEntropy = np.std(originalCrossEntropy)
    stdGeneratedCrossEntropy = np.std(generatedCrossEntropy)

    # Calculate the difference between the original and generated cross-entropy
    crossEntropyDifference = meanGeneratedCrossEntropy - meanOriginalCrossEntropy
    return {
        'SelfCCE-Original': meanOriginalCrossEntropy,
        'SelfCCE-Generated': meanGeneratedCrossEntropy,
        'SelfCCE-Difference': crossEntropyDifference,
        'SelfCCE-Original-Std': stdOriginalCrossEntropy,
        'SelfCCE-Generated-Std': stdGeneratedCrossEntropy
    }
