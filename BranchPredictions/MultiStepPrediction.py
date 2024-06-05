from BranchPredictions import SamplingStrategies
import tqdm, pickle
import torch


class PredictionTreeNode:

    def __init__(self, ids, parent=None, depth=0, beam=0, probs=None, isRandom=False, sparseIndex=None,
                 rootSeqSize=None, discPreds=None, valueHeadData=None):
        self.isRandom = isRandom
        self.parent = parent
        self.children = []
        self.depth = depth
        self.probs = probs
        self.beam = beam
        self.ids = ids
        self.numPredictionIDs = ids.shape[1]
        self.sparseIndex = sparseIndex
        self.rootSeqSize = ids.shape[1] if depth == 0 else rootSeqSize

        self.valueHeadData = valueHeadData
        self.discPreds = discPreds
        self.attMask = None
        self.past = None

    def addGeneratorPredictionToNode(self, ids, past, attMask, probs, isRandomSamples=None):
        self.past = past
        self.attMask = attMask
        isRandomSamples = [False] * ids.shape[1] if isRandomSamples is None else isRandomSamples
        for i in range(ids.shape[1]):
            beam = i if self.parent is None else self.beam
            newNode = PredictionTreeNode(ids[:, i], self, self.depth + 1, beam, probs[:, :, i], isRandomSamples[i],
                                         sparseIndex=self.sparseIndex, rootSeqSize=self.rootSeqSize)
            self.children.append(newNode)

    def addGeneratorPredictionAltsOnlyToNode(self, ids):
        for i in range(ids.shape[1]):
            beam = i if self.parent is None else self.beam
            newNode = PredictionTreeNode(ids[:, i], self, self.depth + 1, beam, sparseIndex=self.sparseIndex,
                                         rootSeqSize=self.rootSeqSize)
            self.children.append(newNode)

    def addDiscPredictionValuesToNode(self, discPreds, valueHeadData=None):
        self.discPreds = discPreds
        self.valueHeadData = valueHeadData

    def addDiscPredictionToNode(self, discPreds, past, attMask, valueHeadData=None):
        self.past = past
        self.attMask = attMask
        self.discPreds = discPreds
        self.valueHeadData = valueHeadData

    def getPaddedPosIDs(self, padValue=9999):
        if (self.depth == 0 or self.ids.shape[1] == self.rootSeqSize):
            return self.sparseIndex + self.depth
        pad = torch.ones((self.rootSeqSize - self.ids.shape[1])).to(self.sparseIndex.device) * padValue
        return torch.cat((self.sparseIndex + self.depth, pad), dim=0)

    def getPaddedAlterationProbs(self, padValue=0.00001):
        if (self.depth == 0 or self.ids.shape[1] == self.rootSeqSize):
            return self.probs

        pad = torch.zeros((self.probs.shape[0], self.rootSeqSize - self.probs.shape[1])).to(
            self.probs.device) + padValue
        return torch.cat((self.probs, pad), dim=1)

    def getLeaves(self, includeRandomNodes=True):
        if (len(self.children) == 0):
            return self
        else:
            results = []
            childLeaves = []
            for c in self.children:
                if (includeRandomNodes or c.isRandom == False):
                    childLeaves.append(c.getLeaves(includeRandomNodes))

            for r in childLeaves:  # Unpack all the results
                if (type(r) == PredictionTreeNode):
                    results.append(r)
                else:
                    for r2 in r:
                        results.append(r2)
            return results

    def getSubNodes(self):
        subNodes = []
        for n in self.children:
            subNodes.append(n)
            subNodes.extend(n.getSubNodes())
        return subNodes

    def getAttentionMask(self, batchSize, device):
        # print(numberOfPredictionIDs, self.numPredictionIDs)
        if (self.depth == 0):
            return None
        if (self.depth == 1):
            if (self.attMask is not None):
                return self.attMask.to(device)

            pastShape = self.parent.past[0][0].shape
            pastSeqLen = pastShape[2]
            attentionMask = torch.zeros((batchSize, self.numPredictionIDs, pastShape[-2] + 1))
            attentionMask[:, :, -1] = 1  # Allow all steps to self-attend
            if (pastSeqLen == len(self.sparseIndex)):
                for i in range(attentionMask.shape[1]):  # Set so all steps attend to all previous steps
                    attentionMask[:, i, :i + 1] = 1
            else:
                for i, s in enumerate(self.sparseIndex):  # Set steps to attend to their individual previous steps
                    attentionMask[:, i, :s + 1] = 1

            return attentionMask.unsqueeze(1).bool().to(device)

        if (self.attMask is not None):
            return self.attMask.to(device)

        # We are at depth >= 2, so we need to merge with previous att masks
        parentMask = self.parent.getAttentionMask(batchSize, device)
        parentMaskNoSelf = parentMask[:, :, :, :-1]

        newAttentionMask = torch.zeros((batchSize, self.numPredictionIDs, self.numPredictionIDs + 1))
        newAttentionMask[:, :, -1] = 1  # Allow all steps to self-attend
        for i in range(self.numPredictionIDs):  # Set so all steps attend to all previous steps
            newAttentionMask[:, i, i] = 1

        # print("Att mask:", parentMaskNoSelf.shape, newAttentionMask.shape)

        return torch.cat((parentMaskNoSelf, newAttentionMask.unsqueeze(1).bool().to(device)), dim=-1)

    def _mergePastWithParent(self, currentMergedPast, parentPast):
        mergedPast = []
        for l1, l2 in zip(parentPast, currentMergedPast):
            k1, v1 = l1
            k2, v2 = l2
            k = torch.concat((k1, k2), dim=-2)
            v = torch.concat((v1, v2), dim=-2)
            mergedPast.append((k, v))

        return mergedPast

    def getFullPast(self):
        if (self.depth == 0):
            return None

        fullPast, n = [], self
        while n.parent is not None:
            # print("Generating Past", n.depth, type(n.parent))
            if (n.past is None):
                fullPast = n.parent.past
            else:
                fullPast = self._mergePastWithParent(fullPast, n.parent.past)
            n = n.parent
        return fullPast

    def producePredictionStep(self, device):
        predIDs = self.ids
        past = self.getFullPast()

        if (self.depth == 0):
            positionalIDs = torch.range(0, predIDs.shape[-1] - 1) + self.depth
        else:
            positionalIDs = self.sparseIndex + self.depth
        positionalIDs = torch.repeat_interleave(positionalIDs.unsqueeze(0), self.ids.shape[0], dim=0)
        positionalIDs = positionalIDs.int()

        attentionMask = self.getAttentionMask(predIDs.shape[0], device)
        return predIDs.to(device), torch.tensor(positionalIDs).to(device), past, attentionMask

    def clearNode(self, clearAll=False, clearAttMask=False, clearSparseIndex=False):
        if (clearAll):
            del self.ids
            self.ids = None

            if (self.discPreds is not None):
                del self.discPreds
            if (self.valueHeadData is not None):
                del self.valueHeadData
            if (self.probs is not None):
                del self.probs

        if ((clearAttMask or clearAll) and self.attMask is not None):
            del self.attMask
            self.attMask = None

        if (self.past is not None):
            del self.past
            self.past = None

        if (clearSparseIndex):
            del self.sparseIndex
            self.sparseIndex = None

    def calculateNumNodes(self):
        if (len(self.children) == 0):
            return 1
        return sum([n.calculateNumNodes() for n in self.children]) + 1

    def moveIDsToDevice(self, device):
        self.ids = self.ids.to(device)
        for n in self.children:
            n.moveIDsToDevice(device)


def clearTree(root, clearAll=False, clearAttMask=False, clearSparseIndex=False):
    def recClear(node):
        for n in node.children:
            recClear(n)
        node.clearNode(clearAll, clearAttMask, clearSparseIndex)

    recClear(root)


def getSubNodeListTreeStructure(root, subNodes):
    for i, n in enumerate(subNodes):
        n.listIndex = i

    alterationLinks = []
    for n in subNodes:
        alterationLinks.append([c.listIndex for c in n.children])

    # Link to all children of root
    rootLinks = []
    for n in root.children:
        if hasattr(n, 'listIndex') and n.listIndex is not None:
            rootLinks.append(n.listIndex)

    return rootLinks, alterationLinks


def getRandomNodesFromRule(strat, params, numAlterations):
    if (strat.name not in SamplingStrategies.RANDOM_TECHNIQUES):
        return None
    if (strat == SamplingStrategies.SamplingTechniques.RANDOM):
        return [True] * numAlterations
    numNonRandomNodes = numAlterations - params['numRandomSamples']
    return [False] * numNonRandomNodes + [True] * params['numRandomSamples']


def _getAltIDsProbs(probs, altIDs):
    exctractionIDs = torch.permute(altIDs, (0, 2, 1))[:, 1:]
    return torch.gather(probs, dim=-1, index=exctractionIDs)


def multiStepGenForward(generator, inputIDs, sampleRules, device, clearTreePast=True, clearFullTree=False,
                        verbose=False, sparseIndex=None):
    inputIDs = inputIDs.to(device)
    firstRule = sampleRules[0]

    firstOutput = generator.forward(
        {'input_ids': inputIDs, 'use_cache': True, 'output_hidden_states': True, 'returnLmLogits': True,
         'dropoutRate': firstRule.samplingParams.get('dropoutRate', None)
         }
    )
    firstTextualPast = firstOutput.prediction.past_key_values
    firstProbs = firstOutput.classProbs

    # Extract probabilities for the IDs and then pad the first timestep
    extractionIDs = inputIDs[:, 1:]
    targetProbs = firstProbs[:, :-1]
    inputProbs = torch.gather(targetProbs, dim=-1, index=extractionIDs.unsqueeze(-1)).squeeze(-1)
    pad = torch.zeros((inputProbs.shape[0], 1)).to(device)
    inputProbs = torch.cat((pad, inputProbs), dim=-1)

    firstRule.samplingParams['includeFinalTimeStep'] = True
    altIDs, maskedProbs = SamplingStrategies.sampleFromGeneratorDistribution(firstProbs, firstRule.numAlterations,
                                                                             inputIDs,
                                                                             firstRule.samplingStrategy,
                                                                             **firstRule.samplingParams)

    # If there's no sparseIndex, we assume all the IDs are valid
    if (sparseIndex is None):
        sparseIndex = torch.range(0, inputIDs.shape[1] - 1).long()

    predictionTreeRoot = PredictionTreeNode(inputIDs, sparseIndex=sparseIndex, probs=inputProbs,
                                            discPreds=firstOutput.discPreds,
                                            )
    altIDsProbs = _getAltIDsProbs(firstProbs, altIDs)
    sparseAltIDsProbs = altIDsProbs[:, sparseIndex]
    sparseAltIDs = altIDs[:, :, 1:][:, :, sparseIndex]

    randomNodeInfo = getRandomNodesFromRule(firstRule.samplingStrategy, firstRule.samplingParams,
                                            firstRule.numAlterations)
    predictionTreeRoot.addGeneratorPredictionToNode(sparseAltIDs, firstTextualPast, None, sparseAltIDsProbs,
                                                    isRandomSamples=randomNodeInfo)

    if (firstOutput.discPreds is not None):
        predictionTreeRoot.addDiscPredictionValuesToNode(firstOutput.discPreds,
                                                         {'InputValue': firstOutput.inputValuePreds,
                                                          'AlterationsValue': firstOutput.alterationsValuePreds}
                                                         )

    for i, rule in tqdm.tqdm(enumerate(sampleRules[1:]), 'Non-Residual Text Generation', disable=not verbose):
        for node in predictionTreeRoot.getLeaves(rule.continueFromRandom):
            predIDs, positionalIDs, past, attMask = node.producePredictionStep(device)

            output = generator.forward(
                {'input_ids': predIDs, 'position_ids': positionalIDs,
                 'textualPast': past, 'customCasualMask': attMask,
                 'output_hidden_states': True, 'returnLmLogits': True, 'use_cache': True,
                 'dropoutRate': rule.samplingParams.get('dropoutRate', None)
                 })

            # Allows one to specify a special rule for beam#0
            if (node.beam == 0 and rule.firstBeamSamplingStrategy is not None):
                strat, params = rule.firstBeamSamplingStrategy, rule.firstBeamSamplingParams
                numAlterations = rule.numAlterations + params['numRandomSamples']
            else:
                strat, params, numAlterations = rule.samplingStrategy, rule.samplingParams, rule.numAlterations

            params['includeFinalTimeStep'] = True
            params['blockSamplingOfLabels'] = False
            labelIDs = torch.concat((predIDs[:, :1], predIDs[:, -1:]), dim=1)
            altIDs, _ = SamplingStrategies.sampleFromGeneratorDistribution(output.classProbs,
                                                                           numAlterations,
                                                                           labelIDs,
                                                                           strat,
                                                                           **params)

            altIDsProbs = _getAltIDsProbs(output.classProbs, altIDs)
            randomNodeInfo = getRandomNodesFromRule(strat, params, numAlterations)
            node.addGeneratorPredictionToNode(altIDs[:, :, 1:], output.prediction.past_key_values, attMask, altIDsProbs,
                                              randomNodeInfo)

            if (output.discPreds is not None):
                node.addDiscPredictionValuesToNode(output.discPreds,
                                                   {'InputValue': output.inputValuePreds,
                                                    'AlterationsValue': output.alterationsValuePreds}
                                                   )

    if (clearTreePast or clearFullTree):
        clearTree(predictionTreeRoot, clearFullTree)

    return predictionTreeRoot, firstOutput.prediction, firstProbs, maskedProbs


def multiStepDiscForward(discriminator, predictionTreeRoot, device, clearTreePast=True, clearFullTree=False,
                         dualValueHeads=False):
    def predNode(node):
        ids, posIDs, past, attMask = node.producePredictionStep(device)
        f = lambda x: x.detach() if x is not None else None
        inData = {'input_ids': f(ids), 'position_ids': f(posIDs), 'textualPast': past,
                  'customCasualMask': f(attMask),
                  'use_cache': True, 'output_hidden_states': True}
        output = discriminator.forward(inData, doDiscForward=False, doTransformerPassWithPrediction=True)
        pred = output.transformerPred
        output.transformerPred = None  # Remove the reference, so that the past can be deleted separately.

        if (dualValueHeads):
            discPreds, inputValuePreds, alterationsValuePreds = discriminator.forward({'pred': pred},
                                                                                      doDiscForward=False,
                                                                                      doFinalizeSinglePrediction=True)
            node.addDiscPredictionToNode(output, pred.past_key_values, attMask,
                                         {'InputValue': inputValuePreds, 'AlterationsValue': alterationsValuePreds}
                                         )
        else:
            node.addDiscPredictionToNode(output, pred.past_key_values, attMask)

    def recPred(node):
        predNode(node)
        for n in node.children:
            recPred(n)

    recPred(predictionTreeRoot)

    if (clearTreePast or clearFullTree):
        clearTree(predictionTreeRoot, clearFullTree, clearAttMask=True, clearSparseIndex=True)
    return predictionTreeRoot
