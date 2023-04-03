from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import LLaMAConfig, LLaMAModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LLaMADecoderLayer

default_llama_config = LLaMAConfig(hidden_size=512,
                                   intermediate_size=512 * 4,
                                   num_hidden_layers=8)

llama_65B_config = LLaMAConfig(hidden_size=8192,
                               num_hidden_layers=80,
                               intermediate_size=22016,
                               num_attention_heads=64)


class LLaMaDecoderLayerWrapper(LLaMADecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:
        return super().forward(hidden_states, attention_mask,
                               output_attentions, use_cache, past_key_value)[0]


class LLamaWrapper(LLaMAModel):
    def __init__(self, config: LLaMAConfig):
        super().__init__(config)
        self.layers = nn.Sequential(*[
            LLaMaDecoderLayerWrapper(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.net = nn.Sequential(
            self.embed_tokens,
            *self.layers,
            self.norm,
        )
        # delattr(self, 'embed_tokens')
        # delattr(self, 'layers')
        # delattr(self, 'norm')

    def forward(
        self,
        input_ids: torch.LongTensor = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        return self.net(input_ids)


def get_llama(hidden_size=512, intermediate_size=512 * 4, num_hidden_layers=1):
    config = LLaMAConfig(hidden_size=hidden_size,
                         intermediate_size=intermediate_size,
                         num_hidden_layers=num_hidden_layers)
    model = LLamaWrapper(config)
    return model


if __name__ == '__main__':
    model = get_llama()
    x = torch.rand([1, 32]).long()
    print(model)
    y = model(x)
    print(y.shape)