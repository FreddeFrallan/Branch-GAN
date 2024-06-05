from Models.GPT2BranchAttention import NonResidualDiscriminatorBase
from Models.NeoXBranchAttention import NeoXNonResidualBase
import torch



class DiscriminatorOutput:

    def __init__(self, prediction=None, classProbs=None, transformerPred=None):
        self.transformerPred = transformerPred
        self.prediction = prediction
        self.classProbs = classProbs


class NonResidualDiscriminator(torch.nn.Module):

    def __init__(self, discriminatorConfig, hasValueHeads=False):
        super().__init__()
        if (discriminatorConfig.model_type == 'gpt_neox'):
            self.nonResidualModel = NeoXNonResidualBase.GPTNeoXForCausalLM(discriminatorConfig)
        else:
            self.nonResidualModel = NonResidualDiscriminatorBase.NonResidualGPT2LMHeadModel(discriminatorConfig)
        self.discriminatorConfig = discriminatorConfig
        self.sigmoidFunc = torch.nn.Sigmoid()

        self.hasValueHeads = hasValueHeads

    def copyTextualPastToFitAlterations(self, textualPast, numAlterationsPerSample):
        if (numAlterationsPerSample <= 1):
            return textualPast

        stackedLayers = []
        for keys, values in textualPast:
            stackedKeys = torch.repeat_interleave(keys, numAlterationsPerSample, dim=0)
            stackedValues = torch.repeat_interleave(values, numAlterationsPerSample, dim=0)
            stackedLayers.append((stackedKeys, stackedValues))

        return stackedLayers

    def forwardSinglePass(self, textualIDs, dropoutRate=None):
        return self.nonResidualModel.forward(textualIDs, output_hidden_states=True, dropoutRate=dropoutRate)

    def _finalizeSingleForwardPredictionPass(self, logits):
        return DiscriminatorOutput(self.sigmoidFunc(logits.squeeze(-1)))

    def forwardPredictionSinglePass(self, ids, dropoutRate=None):
        pred = self.nonResidualModel.forward(ids, output_hidden_states=True, dropoutRate=dropoutRate)
        return self._finalizeSingleForwardPredictionPass(pred.logits)

    def generateTextualPrediction(self, textualIDs):
        textualPrediction = self.nonResidualModel.forward(textualIDs, use_cache=True,
                                                          output_hidden_states=self.hasValueHeads)
        textualPast = textualPrediction['past_key_values']
        return textualPrediction, textualPast

    def generateAlterationPrediction(self, textualPast, alternationIDs):
        stackedAlternationIDs = torch.reshape(
            alternationIDs,
            (alternationIDs.shape[0] * alternationIDs.shape[1], alternationIDs.shape[2])
        )

        textualPast = self.copyTextualPastToFitAlterations(textualPast, alternationIDs.shape[1])
        nonResPrediction = self.nonResidualModel(stackedAlternationIDs, textualPast=textualPast,
                                                 output_hidden_states=self.hasValueHeads)

        return nonResPrediction

    def finalizeDiscForward(self, textualPrediction, nonResPrediction, alternationIDs):
        alternationLogits = torch.reshape(nonResPrediction.logits, alternationIDs.shape)
        textPred, altPred = self.sigmoidFunc(textualPrediction.logits.squeeze(-1)), self.sigmoidFunc(alternationLogits)
        return DiscriminatorOutput(textPred), DiscriminatorOutput(altPred)

    def discForward(self, textualIDs, alternationIDs):
        # Textual IDs shape -> (numSamples, textLength)
        # Alternation ids shape -> (numSamples, numAlterationsPerSample, textLength)

        textualPrediction, textualPast = self.generateTextualPrediction(textualIDs)
        nonResPrediction = self.generateAlterationPrediction(textualPast, alternationIDs)
        return self.finalizeDiscForward(textualPrediction, nonResPrediction, alternationIDs)

    def forward(self, inputArgs, doDiscForward=True, doTransformerPass=False, doFinalizeSinglePrediction=False,
                doTransformerPassWithPrediction=False):
        if (doDiscForward):
            return self.discForward(**inputArgs)
        elif (doTransformerPass):
            return self.nonResidualModel.forward(**inputArgs)
        elif (doTransformerPassWithPrediction):
            transPred = self.nonResidualModel.forward(**inputArgs)
            output = self._finalizeSingleForwardPredictionPass(transPred.logits)
            output.transformerPred = transPred
            return output
        elif (doFinalizeSinglePrediction):
            return self._finalizeSinglePrediction(**inputArgs)

class NonResidualDiscriminatorWithDualValueHeads(NonResidualDiscriminator):

    def __init__(self, discriminatorConfig):
        super().__init__(discriminatorConfig, hasValueHeads=True)
        self.inputValueHead = torch.nn.Linear(discriminatorConfig.hidden_size, 1)
        self.alterationsValueHead = torch.nn.Linear(discriminatorConfig.hidden_size, 1)

    def _finalizeSinglePrediction(self, pred):
        alterationsValueHeadLogits = self.alterationsValueHead(pred.hidden_states[-1])
        inputValueHeadLogits = self.inputValueHead(pred.hidden_states[-1])

        alterationsValuePreds = self.sigmoidFunc(alterationsValueHeadLogits)
        inputValuePreds = self.sigmoidFunc(inputValueHeadLogits)
        discPreds = self.sigmoidFunc(pred.logits.squeeze(-1))

        return discPreds, inputValuePreds, alterationsValuePreds

    def finalizeDiscForward(self, textualPrediction, nonResPrediction, alternationIDs):
        txtDiscPreds, txtInputValuePreds, txtAlterationsValuePreds = self._finalizeSinglePrediction(textualPrediction)
        _, altInputValuePreds, altAlterationsValuePreds = self._finalizeSinglePrediction(nonResPrediction)

        # Unstack the alteration predictions
        altAlterationsValuePreds = torch.reshape(altAlterationsValuePreds, alternationIDs.shape)
        altInputValuePreds = torch.reshape(altInputValuePreds, alternationIDs.shape)

        # This is done here in order to be compatible with the current shapes of the altDiscPredictions
        alternationLogits = torch.reshape(nonResPrediction.logits, alternationIDs.shape)
        altDiscPreds = self.sigmoidFunc(alternationLogits)

        return txtDiscPreds, altDiscPreds, {'txtInputPreds': txtInputValuePreds,
                                            'txtAlterationsPreds': txtAlterationsValuePreds,
                                            'altInputPreds': altInputValuePreds,
                                            'altAlterationsPreds': altAlterationsValuePreds}


