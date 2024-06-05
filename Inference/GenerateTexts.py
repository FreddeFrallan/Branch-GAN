import torch
import tqdm


def generateTexts(generator, data, settings, device, tokenizer=None, generationLength=128, startIndex=None,
                  maxNumSamples=None, verbose=False):
    collectedResults, sampleCounter = [], 0

    # If maxNumSamples is given we use a custom progress bar
    if (maxNumSamples is not None):
        pBar = tqdm.tqdm(total=maxNumSamples, desc='Generating Texts', disable=not verbose)

    with torch.no_grad():
        for ids in tqdm.tqdm(data, 'Generating Texts', disable=(maxNumSamples is not None or not verbose)):
            bIDs = torch.tensor(ids).to(device)
            bIDs = bIDs if startIndex is None else bIDs[:, startIndex:]

            outputSeqs = generator.generate(bIDs, max_new_tokens=generationLength, **settings, pad_token_id=0)
            generatedSeqs = outputSeqs[:, bIDs.shape[1]:]

            for c, g in zip(bIDs, generatedSeqs):
                collectedResults.append((c.tolist(), g.tolist()))

            if (maxNumSamples is not None):
                sampleCounter += len(ids)
                pBar.update(len(ids))
                if (sampleCounter >= maxNumSamples):
                    break

    if (tokenizer is None):
        return collectedResults, None

    collectedTexts = []
    for context, generated in collectedResults:
        contextText = tokenizer.decode(context)
        generatedText = tokenizer.decode(generated)
        collectedTexts.append((contextText, generatedText))
    return collectedResults, collectedTexts


def calculateSelfCrossEntropy(generator, device, data, generatedSeqs, verbose=False):
    contextLength, genLength = len(generatedSeqs[0][0]), len(generatedSeqs[0][1])
    stackedGenSeqs = torch.tensor([s[0] + s[1] for s in generatedSeqs])
    numSamples = len(generatedSeqs)
    batchSize = None

    def logitsToLoss(logits, ids):
        # Calculate the full loss, return the loss for the generated part only
        probs = torch.softmax(logits, dim=-1)
        targetProbs = torch.gather(probs[:, :-1], -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        lossGenerated = -torch.log(targetProbs[:, contextLength:])
        return lossGenerated.mean(-1)

    originalCrossEntropy, generatedCrossEntropy = [], []
    with torch.no_grad():
        sampleCounter = 0
        for bIDs in tqdm.tqdm(data, 'Calculating Original Cross-Entropy', disable=not verbose):
            bIDs = torch.tensor(bIDs[:, :contextLength + genLength]).to(device)
            batchSize = bIDs.shape[0] if batchSize is None else batchSize

            output = generator(bIDs)
            loss = logitsToLoss(output.logits, bIDs)
            for i in range(loss.shape[0]):
                originalCrossEntropy.append(loss[i].item())
                sampleCounter += 1
                if (sampleCounter >= numSamples):
                    break

            if (sampleCounter >= numSamples):
                break

        for i in tqdm.tqdm(range(0, numSamples, batchSize), 'Calculating Generated Cross-Entropy', disable=not verbose):
            bIDs = stackedGenSeqs[i:i + batchSize].to(device)
            output = generator(bIDs)
            loss = logitsToLoss(output.logits, bIDs)
            generatedCrossEntropy.extend(loss.tolist())

    return originalCrossEntropy, generatedCrossEntropy


def main():
    from Training.Metrics import GeneratedTextsMetric
    from DataManagers import DataUtils
    from Models import FileManagement
    import transformers

    tokenizerPath = 'EleutherAI/pythia-410m-deduped'
    device = 'cuda:5'
    batchSize = 16

    # savePath = 'OriginalPythia-410m-Greedy-Wiki-Texts'
    # modelPath = "EleutherAI/pythia-410m-deduped"

    # modelPath = 'Single-MLE+Disc-NoPreTraining-410M-B16-1706969119.8060038'
    # savePath = 'OwnMLE-Greedy-Sample-Wiki-Texts'
    # checkpoint = 'Epoch-4'

    # modelPath = 'Single-410M-B16-D16-A32-Lr1e-5-1707669410.6424696'
    # savePath = 'Single-PythiaStart-410m-Sample-Texts'
    # checkpoint = 'Epoch-8'

    modelPath = 'Single-OwnMLEPreTraining-410M-B16-D16-A32-1707514905.5823448'
    savePath = 'Single-MleStart-410m-Greedy-Texts'
    checkpoint = 'Epoch-8'

    # Load the model
    model = FileManagement.loadModel(modelPath, checkpoint).generator.generator.to(device)
    # model = transformers.AutoModelForCausalLM.from_pretrained(modelPath).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizerPath)

    testDataloader = DataUtils.loadPreTokenizedDataloader(batchSize, 'Wikitext_Pythia_256',
                                                          maxNumSamples=128, isValidation=True, doShuffle=False,
                                                          returnTorchDataLoader=True
                                                          )

    settings = {
        'temperature': 1.0,
        'num_beams': 1,
        'do_sample': False,
        'num_return_sequences': 1
    }

    results, texts = generateTexts(model,
                                   data=testDataloader, settings=settings, tokenizer=tokenizer, device=device,
                                   startIndex=128, generationLength=96, verbose=True
                                   )
    for context, generated in texts:
        print("*" * 80)
        print(f"Context:\n {context}")
        print(f"Generated:\n {generated}")
        print()

    TTR, contextTTR = GeneratedTextsMetric.calculateTTR(results)
    print(f"Total TTR: {TTR}")
    print(f"Context TTR: {contextTTR}")

    original, generated = calculateSelfCrossEntropy(model, device, testDataloader, results, verbose=True)
    selfCceResults = GeneratedTextsMetric.selfCrossEntropy(original, generated)
    selfCceResults.update({'TTR': TTR, 'ContextTTR': contextTTR})

    saveData = {
        'Model': modelPath.replace('/', '-'),
        'Settings': settings,
        'Results': selfCceResults,
        'Texts': [{'Context': context, 'Generated': generated} for context, generated in texts],
    }

    import json
    with open(savePath, 'w') as fp:
        json.dump(saveData, fp)
