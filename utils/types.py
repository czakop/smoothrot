from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
)
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralPreTrainedModel,
    MistralRMSNorm,
)

MODEL_TYPE = LlamaPreTrainedModel | MistralPreTrainedModel
DECODER_LAYER_TYPE = LlamaDecoderLayer | MistralDecoderLayer
LAYERNORM_TYPE = LlamaRMSNorm | MistralRMSNorm
