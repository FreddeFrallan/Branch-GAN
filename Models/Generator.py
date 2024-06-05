from Models.GPT2BranchAttention import NonResidualDiscriminatorBase
from Models.NeoXBranchAttention import NeoXNonResidualBase
import transformers
import torch

class GeneratorOutput:

    def __init__(self, prediction=None, classProbs=None, discPreds=None, inputValuePreds=None, alterationsValuePreds=None):
        self.prediction = prediction
        self.classProbs = classProbs
        self.discPreds = discPreds
        self.inputValuePreds = inputValuePreds
        self.alterationsValuePreds = alterationsValuePreds

class Generator(torch.nn.Module):

    def __init__(self, generatorConfig):
        super().__init__()
        self.config = generatorConfig
        self.generator = self.createGenerator(generatorConfig)

        self.useCache = False

    def createGenerator(self, generatorConfig):
        if (generatorConfig['model_type'] == 'gpt_neox'):
            return NeoXNonResidualBase.GPTNeoXForCausalLM(generatorConfig)
        return transformers.GPT2LMHeadModel(generatorConfig)

    def forward(self, inputIDs):
        prediction = self.generator.forward(inputIDs, output_hidden_states=True, use_cache=self.useCache)
        probs = torch.softmax(prediction.logits, dim=-1)

        return GeneratorOutput(prediction, probs)


class NonResidualGenerator(Generator):

    def createGenerator(self, generatorConfig):
        if (generatorConfig.model_type == 'gpt_neox'):
            return NeoXNonResidualBase.GPTNeoXForCausalLM(generatorConfig)
        return NonResidualDiscriminatorBase.NonResidualGPT2LMHeadModel(generatorConfig)

    def forward(self, args):
        args['output_hidden_states'] = True

        prediction = self.generator.forward(**args)
        probs = torch.softmax(prediction.logits, dim=-1)

        return GeneratorOutput(prediction, probs)

