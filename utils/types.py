from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
)
from transformers.models.mistral.modeling_mistral import (
    MistralConfig,
    MistralDecoderLayer,
    MistralPreTrainedModel,
    MistralRMSNorm,
)

MODEL_TYPE = LlamaPreTrainedModel | MistralPreTrainedModel
DECODER_LAYER_TYPE = LlamaDecoderLayer | MistralDecoderLayer
LAYERNORM_TYPE = LlamaRMSNorm | MistralRMSNorm
CONFIG_TYPE = LlamaConfig | MistralConfig
