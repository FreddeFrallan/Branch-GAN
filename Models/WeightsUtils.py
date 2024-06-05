from Models import FileManagement
import transformers

def createNonResModelFromHFModel(modelPath, modelSavePath, generatorClass=None, discriminatorClass=None):
    from Models import Discriminator, Generator

    if (generatorClass is None):
        print("No generator class provided, using default non-residual generator")
        generatorClass = Generator.NonResidualGenerator
    if (discriminatorClass is None):
        print("No discriminator class provided, using default non-residual discriminator")
        discriminatorClass = Discriminator.NonResidualDiscriminatorWithDualValueHeads

    print("Loading HF Model")
    hfModel = transformers.AutoModelForCausalLM.from_pretrained(modelPath)
    config = transformers.AutoConfig.from_pretrained(modelPath)

    print("Loading Discriminator")
    print("Creating discriminator with dual value heads")
    discriminator = discriminatorClass(config)

    print("Loading Generator")
    generator = generatorClass(config)

    print("Transferring weights to Discriminator and Generator")
    discriminator_state_dict = discriminator.state_dict()
    generator_state_dict = generator.state_dict()

    hf_state_dict = hfModel.state_dict()

    numGeneratorWeights, numDiscriminatorWeights = 0, 0
    for k, v in hf_state_dict.items():
        for dk, dv in discriminator_state_dict.items():
            if k in dk:
                discriminator_state_dict[dk] = v
                numDiscriminatorWeights += 1
        for gk, gv in generator_state_dict.items():
            if k in gk:
                generator_state_dict[gk] = v
                numGeneratorWeights += 1

    discriminator.load_state_dict(discriminator_state_dict)
    generator.load_state_dict(generator_state_dict)

    print("Weights transferred successfully")
    print("Number of weights transferred to discriminator: {}".format(numDiscriminatorWeights))
    print("Number of weights transferred to generator: {}".format(numGeneratorWeights))
    print("Number of total weights in discriminator: {}".format(len(discriminator_state_dict)))
    print("Number of total weights in generator: {}".format(len(generator_state_dict)))

    class GANNModel:
        def __init__(self, discriminator, generator, discriminatorConfig, generatorConfig):
            self.discriminator = discriminator
            self.generator = generator
            self.discriminatorConfig = discriminatorConfig
            self.generatorConfig = generatorConfig

    model = GANNModel(discriminator, generator, config, config)

    # Save the model using saveGannSetup function
    FileManagement.saveGannSetup(model, modelSavePath, 'Start')