import transformers.models.gpt2.modeling_gpt2
import transformers
import torch

from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)

from packaging import version

if version.parse(torch.__version__) >= version.parse("1.6"):
    is_amp_available = True
    from torch.cuda.amp import autocast
else:
    is_amp_available = False


class GPT2NonResidualAttention(torch.nn.Module):

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer(
            "self_bias",
            torch.ones((max_positions, max_positions), dtype=torch.uint8).view(1, 1, max_positions, max_positions),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = torch.nn.Dropout(config.attn_pdrop)
        self.resid_dropout = torch.nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def getCausualAttentionMask(self, query_length, key_length, nonResidual=False):
        if (nonResidual):
            nonResidualCasual = self.bias[:, :, key_length - query_length: key_length, 1:key_length + 1].bool()
            selfMask = self.self_bias[:, :, key_length - query_length: key_length, :1].bool()
            return torch.cat((nonResidualCasual, selfMask), dim=-1)
        else:
            return self.bias[:, :, key_length - query_length: key_length, :key_length].bool()

    def _attn(self, query, key, value, attention_mask=None, head_mask=None, promptModelMask=None, dropoutRate=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (value.size(-1) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            if (promptModelMask is None):
                attMask = self.getCausualAttentionMask(query_length, key_length)
            else:
                attMask = promptModelMask
            #     print("Using supplied promptModelMask")
            # print(attMask.shape)
            attn_weights = torch.where(attMask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        # attn_weights = self.attn_dropout(attn_weights)
        attn_weights = torch.nn.functional.dropout(attn_weights,
                                                   dropoutRate if dropoutRate is not None else self.config.attn_pdrop)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def nonResidualTextAttention(self, query, key, value, textualKeyPast, textualValuePast, head_mask=None,
                                 customCasualMask=None, dropoutRate=None):
        textWeights = torch.matmul(query, textualKeyPast.transpose(-1, -2))
        selfWeights = torch.sum(query * key, keepdim=True, dim=-1)

        attn_weights = torch.cat((textWeights, selfWeights), dim=-1)

        if self.scale_attn_weights:
            attn_weights = attn_weights / (textualValuePast.size(-1) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            batchSize, query_length, key_length = query.size(0), query.size(-2), textualKeyPast.size(-2)
            # print("Attention Sizes:", batchSize, query_length, key_length)
            if (customCasualMask is None):
                causal_mask = self.getCausualAttentionMask(query_length, key_length, nonResidual=True)
                sampleCausalMask = causal_mask.repeat_interleave(batchSize, dim=0)
            else:
                sampleCausalMask = customCasualMask
            # print("Sample Causal Mask:", sampleCausalMask.shape)
            # print(sampleCausalMask[0].int())
            attentionMask = sampleCausalMask
            # print("Attention Mask:", attentionMask.shape)
            # print("Attention Weight", attn_weights.shape)
            attn_weights = torch.where(attentionMask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        # if promptMask is not None:
        #     # Apply the attention mask
        #     attn_weights = attn_weights + promptMask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(textualValuePast.dtype)
        # attn_weights = self.attn_dropout(attn_weights)
        attn_weights = torch.nn.functional.dropout(attn_weights,
                                                   dropoutRate if dropoutRate is not None else self.config.attn_pdrop)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        wSelf, wRest = attn_weights[:, :, :, -1:], attn_weights[:, :, :, :-1]

        valuePast = textualValuePast
        outRest = torch.matmul(wRest, valuePast)
        outSelf = wSelf * value
        attn_output = outRest + outSelf

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, use_cache=False, output_attentions=False, nonResidual=False,
                nonResidualTextOnly=False, nonResdiaulPastOnly=False, promptPast=None, textualPast=None,
                promptMask=None, customCasualMask=None, dropoutRate=None):
        # print("Dropout Attention", dropoutRate is not None)
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        # print(self.layer_idx, torch.sum(key, dim=(0, 1, 3)))

        if ((nonResidual == False and nonResidualTextOnly == False) or nonResdiaulPastOnly):
            if (nonResdiaulPastOnly):
                # import DebugUtils
                layer_past = promptPast
                # print("Prompt mask:", promptMask.shape)
                query_length, key_length = query.size(-2), key.size(-2)
                if (customCasualMask is None):
                    causualMask = self.getCausualAttentionMask(query_length, key_length)
                    causualMask = torch.repeat_interleave(causualMask, query.shape[0], dim=0)
                else:
                    causualMask = customCasualMask

                # print(causualMask.shape)
                # promptMask = promptMask.int() * 0
                # promptMask = promptMask.bool()
                promptMask = torch.cat((promptMask, causualMask), dim=-1)
                # rows = [list(x) for x in promptMask[0, 0].int().numpy()]
                # DebugUtils.createRowExcelFile('Debug PromptMask.xlsx', rows)
                # print("Prompt Mask", promptMask.shape)
                # input()

                # print("Prompt mask post:", promptMask.shape)

            if layer_past is not None:
                past_key, past_value = layer_past
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)

            # print(query.shape, key.shape, value.shape)

            if use_cache is True:
                present = (key, value)
            else:
                present = None

            # print(self.layer_idx, torch.sum(key, dim=(0, 1, 3)))
            if self.reorder_and_upcast_attn:
                attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask,
                                                                            head_mask)
            else:
                attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask,
                                                       promptModelMask=promptMask, dropoutRate=dropoutRate)
        elif (nonResidualTextOnly):
            if use_cache is True:
                present = (key, value)
            else:
                present = None

            textualKey, textualValue = textualPast
            attn_output, attn_weights = self.nonResidualTextAttention(query, key, value, textualKey, textualValue,
                                                                      head_mask, customCasualMask=customCasualMask,
                                                                      dropoutRate=dropoutRate)
        else:
            raise NotImplementedError()

        # print("Attention output:", attn_output.shape)
        # print("Num heads:", self.num_heads)
        # print("Num Dims:", self.head_dim)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        # attn_output = self.resid_dropout(attn_output)
        attn_output = torch.nn.functional.dropout(attn_output,
                                                  dropoutRate if dropoutRate is not None else self.config.resid_pdrop)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2NonResidualAttentionWithFutureInstructionQueue(GPT2NonResidualAttention):

    def __init__(self, config, is_cross_attention=False, layer_idx=None, numberOfFutureInstructions=1):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.numberOfFutureInstructions = numberOfFutureInstructions
        self.futureInstructionKeyTransformations = [
            torch.nn.Linear(config.hidden_size, config.hidden_size) for _ in range(numberOfFutureInstructions)
        ]

        self.futureInstructionValueTransformations = [
            torch.nn.Linear(config.hidden_size, config.hidden_size) for _ in range(numberOfFutureInstructions)
        ]

        for i, transformation in enumerate(self.futureInstructionKeyTransformations):
            for j, p in enumerate(transformation.parameters()):
                self.register_parameter('FutureKey-{}-{}'.format(i, j), p)
        for i, transformation in enumerate(self.futureInstructionValueTransformations):
            for j, p in enumerate(transformation.parameters()):
                self.register_parameter('FutureValue-{}-{}'.format(i, j), p)

    def getFutureKeyValues(self, promptKeyPast, promptValuePast):
        futureKeys = [trans(promptKeyPast) for trans in self.futureInstructionKeyTransformations]
        futureValues = [trans(promptValuePast) for trans in self.futureInstructionValueTransformations]
        stackedKeys = torch.stack([promptKeyPast] + futureKeys, dim=-3)
        stackedValues = torch.stack([promptValuePast] + futureValues, dim=-3)
        return stackedKeys, stackedValues

    def expandValueToMatchFutureQ(self, data):
        return torch.repeat_interleave(data.unsqueeze(dim=-3), self.numberOfFutureInstructions + 1, dim=-3)

    def maskAndReduce(self, data, mask):
        mask = mask.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
        maskedData = data * mask
        return torch.sum(maskedData, dim=-3)

    def nonResidualAttention(self, query, key, value, promptKeyPast, promptValuePast, textualKeyPast, textualValuePast,
                             promptMask=None, head_mask=None, futureInstructionMask=None, dropoutRate=None):
        stackedPromptKeys, stackedPromptValues = self.getFutureKeyValues(promptKeyPast, promptValuePast)
        expandedQueries = self.expandValueToMatchFutureQ(query)
        print("Stacked Prompt keys and values", stackedPromptKeys.shape, stackedPromptValues.shape)
        print("Expanded Queries", expandedQueries.shape)

        promptWeights = torch.matmul(expandedQueries, stackedPromptKeys.transpose(-1, -2))
        print("Prompt weights", promptWeights.shape)
        reducedPromptWeights = self.maskAndReduce(promptWeights, futureInstructionMask)
        print("ReducedPromptWeights", reducedPromptWeights.shape)

        textWeights = torch.matmul(query, textualKeyPast.transpose(-1, -2))
        selfWeights = torch.sum(query * key, keepdim=True, dim=-1)

        attn_weights = torch.cat((reducedPromptWeights, textWeights, selfWeights), dim=-1)

        if self.scale_attn_weights:
            attn_weights = attn_weights / (textualValuePast.size(-1) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            batchSize, query_length, key_length = query.size(0), query.size(-2), textualKeyPast.size(-2)
            causal_mask = self.getCausualAttentionMask(query_length, key_length, nonResidual=True)
            sampleCausalMask = causal_mask.repeat_interleave(batchSize, dim=0)
            # print(sampleCausalMask.shape, promptMask.shape)
            attentionMask = torch.cat((promptMask, sampleCausalMask), dim=-1)
            attn_weights = torch.where(attentionMask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        # if promptMask is not None:
        #     # Apply the attention mask
        #     attn_weights = attn_weights + promptMask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(textualValuePast.dtype)
        # attn_weights = self.attn_dropout(attn_weights)
        attn_weights = torch.nn.functional.dropout(attn_weights,
                                                   dropoutRate if dropoutRate is not None else self.config.attn_pdrop)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        wSelf, wRest = attn_weights[:, :, :, -1:], attn_weights[:, :, :, :-1]

        valuePast = torch.cat((promptValuePast, textualValuePast), dim=-2)
        outRest = torch.matmul(wRest, valuePast)
        outSelf = wSelf * value
        attn_output = outRest + outSelf

        return attn_output, attn_weights


def sanityCheckCasualMask():
    config = transformers.GPT2Config()
    attention = GPT2NonResidualAttention(config, is_cross_attention=False)
    mask = attention.getCausualAttentionMask(5, 5, nonResidual=True)
    print(mask)
    print(mask.shape)


def main():
    sanityCheckCasualMask()
