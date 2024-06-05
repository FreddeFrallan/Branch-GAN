from transformers.models.gpt2 import modeling_gpt2
import torch

def mleLossFromProbs(inputIDs, generatorProbs, aggrigate=True):
    selectedProbs = torch.gather(generatorProbs[:, :-1], dim=-1, index=inputIDs[:, 1:].unsqueeze(-1))
    logProbs = torch.log(selectedProbs)
    if (aggrigate):
        return -logProbs.mean()
    return -logProbs


def mleLossFromLogits(inputIDs, logits):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputIDs[..., 1:].contiguous()
    loss_fct = modeling_gpt2.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))