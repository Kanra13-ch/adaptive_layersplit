from transformers import AutoTokenizer, LlamaForCausalLM, LlamaModel
import transformers
import time
import random
import torch
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from datasets import load_dataset
from tqdm import tqdm

logger = logging.get_logger(__name__)
class ModifiedLlamaModel(LlamaModel):
    def __init__(self, config):
        super(ModifiedLlamaModel, self).__init__(config)
        self.dropout_prob = 0.1
        self.n = 3
        self.delay_range = (0.005, 0.01)

    def set_dropout_prob(self, dropout_prob):
        self.dropout_prob = dropout_prob

    def set_specified_layers(self, specified_layers):
        self.n = specified_layers

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
            # print(i)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if i == self.n:
                # print(hidden_states)
                # hidden_states = self.add_noise(hidden_states)
                hidden_states = self.probabilistic_signal_dropout(hidden_states)
                # print(self.n)
                self.simulate_delay()

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    def add_noise(self, hidden_states):
        noise = torch.rand_like(hidden_states) * self.noise_level
        # print("noise\n -------")
        return hidden_states + noise

    def probabilistic_signal_dropout(self, hidden_states):
        """
        Apply probabilistic signal dropout to the hidden states.
        Each element of the hidden states has a `dropout_prob` chance of being set to zero.

        :param hidden_states: torch.Tensor - The hidden states of the model.
        :return: torch.Tensor - Modified hidden states.
        """
        rand_matrix = torch.rand(hidden_states.shape, device=hidden_states.device)
        dropout_mask = rand_matrix < self.dropout_prob
        # print(rand_matrix)
        return hidden_states * ~dropout_mask

    def simulate_delay(self):
        # print("delay\n -------")
        delay_time = random.uniform(*self.delay_range)
        # time.sleep(delay_time)

class LlamaForCausalLMWithChannel(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = ModifiedLlamaModel(config)

    def set_dropout_prob(self, dropout_prob):
        self.model.set_dropout_prob(dropout_prob)

    def set_specified_layers(self, n):
        self.model.set_specified_layers(n)

model_id = "./Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = LlamaForCausalLMWithChannel.from_pretrained(model_id,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             )
# model.set_dropout_prob(0.3)
# model.set_specified_layers(8)
device = "cuda"

def infer(inp):
    input_ids = tokenizer(inp, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    generate_input = {
        "input_ids":input_ids,
        "max_new_tokens":64,
        "do_sample":True,
        "top_k":40,
        "top_p": 0.95,
        "temperature": 0.3,
        "num_beams": 3,
        "repetition_penalty": 1.3,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    outputs = model.generate(**generate_input)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return response
def calculate_ppl(model, tokenizer, device, num_runs=10):
    # test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test = load_dataset('parquet', data_files='test.parquet', split="train")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    max_length = model.config.max_length
    stride = 512
    seq_len = encodings.input_ids.size(1)

    avg_ppl = 0
    for _ in tqdm(range(num_runs)):
        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
            if not torch.isnan(neg_log_likelihood):
                nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        avg_ppl += ppl.item()
    avg_ppl /= num_runs
    return avg_ppl

def log_results(filename, dropout_prob, n, avg_ppl):
    with open(filename, "a") as file:
        file.write(f"dropout_prob={dropout_prob}, n={n}, ppl={avg_ppl}\n")

def dropout_prob_generator(start, end, step):
    while start <= end:
        yield start
        start += step

# dropout_prob = 0.02
# n = 1
log_filename = "experiment_results.txt"
for dropout_prob in tqdm(dropout_prob_generator(0.02, 0.3, 0.04), desc="Progress of dropout_prob"):
    model.set_dropout_prob(dropout_prob)
    for n in tqdm(range(1, 10), desc='Progress of n', leave=False):
        model.set_specified_layers(n)
        avg_ppl = calculate_ppl(model, tokenizer, device, 5)
        log_results(log_filename, dropout_prob, n, avg_ppl)

