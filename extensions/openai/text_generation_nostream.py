import ast
import html
import time
import traceback
import json

import torch
import transformers
from transformers import LogitsProcessorList

import modules.shared as shared
from modules.callbacks import (
    Iteratorize,
    Stream,
    _StopEverythingStoppingCriteria
)
from modules.extensions import apply_extensions
from modules.logging_colors import logger
from modules.models import clear_torch_cache, local_rank
from modules.text_generation import encode, decode, get_encoded_length, get_token_ids, get_max_prompt_length, generate_reply_wrapper, formatted_outputs, \
    fix_gpt4chan, fix_galactica, set_manual_seed, stop_everything_event, apply_stopping_strings

def get_reply_from_output_ids(output_ids, input_ids, original_question, state, is_chat=False):
    if shared.is_seq2seq:
        reply = decode(output_ids, state['skip_special_tokens'])
    else:
        new_tokens = len(output_ids) - len(input_ids[0])
        reply = decode(output_ids[-new_tokens:], state['skip_special_tokens'])
        # Prevent LlamaTokenizer from skipping a space
        if type(shared.tokenizer) in [transformers.LlamaTokenizer, transformers.LlamaTokenizerFast] and len(output_ids) > 0:
            if shared.tokenizer.convert_ids_to_tokens(int(output_ids[-new_tokens])).startswith('▁'):
                reply = ' ' + reply

    return reply

def generate_reply(*args, **kwargs):
    replies = []
    try:
        shared.generation_lock.acquire()
        results = _generate_reply(*args, **kwargs)
        for result in results:
            replies.append(result)
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
    finally:
        shared.generation_lock.release()
        return replies


def _generate_reply(question, state, stopping_strings=None, is_chat=False, escape_html=False):
    replies = []
    # Find the appropriate generation function
    generate_func = apply_extensions('custom_generate_reply')
    if generate_func is None:
        if shared.model_name == 'None' or shared.model is None:
            logger.error("No model is loaded! Select one in the Model tab.")
            return replies

        if shared.model.__class__.__name__ in ['LlamaCppModel', 'RWKVModel', 'ExllamaModel', 'Exllamav2Model', 'CtransformersModel']:
            generate_func = generate_reply_custom
        else:
            generate_func = generate_reply_HF

    # Prepare the input
    original_question = question
    state = apply_extensions('state', state)
    question = apply_extensions('input', question, state)

    # Find the stopping strings
    all_stop_strings = []
    for st in (stopping_strings, ast.literal_eval(f"[{state['custom_stopping_strings']}]")):
        if type(st) is list and len(st) > 0:
            all_stop_strings += st

    if shared.args.verbose:
        print(f'\n\n{question}\n--------------------\n')

    shared.stop_everything = False
    clear_torch_cache()
    seed = set_manual_seed(state['seed'])

    results = generate_func(question, original_question, seed, state, stopping_strings, is_chat=is_chat)
    for result in results:
        if escape_html:
            result = html.escape(result)
        replies.append(result)

    return replies

def generate_reply_HF(question, original_question, seed, state, stopping_strings=None, is_chat=False):
    replies = []  # Initialize an empty list to store generated replies
    generate_params = {}
    for k in ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty',
              'repetition_penalty_range', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha',
              'length_penalty', 'early_stopping', 'tfs', 'top_a', 'mirostat_mode', 'mirostat_tau', 'mirostat_eta', 'guidance_scale']:
        if k in state:
            generate_params[k] = state[k]

    if 'num_return_sequences' in state:
        generate_params['num_return_sequences'] = state['num_return_sequences']
    else:
        generate_params['num_return_sequences'] = 1

    if state['negative_prompt'] != '':
        generate_params['negative_prompt_ids'] = encode(state['negative_prompt'])

    for k in ['epsilon_cutoff', 'eta_cutoff']:
        if state[k] > 0:
            generate_params[k] = state[k] * 1e-4

    if state['ban_eos_token']:
        generate_params['suppress_tokens'] = [shared.tokenizer.eos_token_id]

    if state['custom_token_bans']:
        to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
        if len(to_ban) > 0:
            if generate_params.get('suppress_tokens', None):
                generate_params['suppress_tokens'] += to_ban
            else:
                generate_params['suppress_tokens'] = to_ban

    generate_params.update({'use_cache': not shared.args.no_cache})
    if shared.args.deepspeed:
        generate_params.update({'synced_gpus': True})

    # Encode the input
    input_ids = encode(question, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
    cuda = not any((shared.args.cpu, shared.args.deepspeed))
    if state['auto_max_new_tokens']:
        generate_params['max_new_tokens'] = state['truncation_length'] - input_ids.shape[-1]

    # Add the encoded tokens to generate_params
    question, input_ids, inputs_embeds = apply_extensions('tokenizer', state, question, input_ids, None)
    original_input_ids = input_ids
    generate_params.update({'inputs': input_ids})
    if inputs_embeds is not None:
        generate_params.update({'inputs_embeds': inputs_embeds})

    # Stopping criteria / eos token
    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    generate_params['eos_token_id'] = eos_token_ids
    generate_params['stopping_criteria'] = transformers.StoppingCriteriaList()
    generate_params['stopping_criteria'].append(_StopEverythingStoppingCriteria())

    processor = state.get('logits_processor', LogitsProcessorList([]))
    # In case folks just pass in a processor by itself.
    if type(processor) != LogitsProcessorList:
        processor = LogitsProcessorList([processor])
    apply_extensions('logits_processor', processor, input_ids)
    generate_params['logits_processor'] = processor

    t0 = time.time()
    original_tokens = len(original_input_ids[0])
    new_tokens = 0
    try:
        with torch.no_grad():
            outputs = shared.model.generate(**generate_params)
            if cuda:
                outputs = outputs.cuda()
            for output in outputs:
                new_tokens += len(output) - (original_tokens if not shared.is_seq2seq else 0)
                reply = get_reply_from_output_ids(output, input_ids, original_question, state, is_chat=is_chat)
                replies.append(reply)
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
    finally:
        t1 = time.time()
        print(f'Output generated in {(t1 - t0):.2f} seconds ({new_tokens / (t1 - t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
    return replies


def generate_reply_custom(question, original_question, seed, state, stopping_strings=None, is_chat=False):
    """
    For models that do not use the transformers library for sampling
    """
    replies = []  # Initialize an empty list to store generated replies
    #    seed = set_manual_seed(state['seed'])

    if not 'num_return_sequences' in state:
        state['num_return_sequences'] = 1

    t0 = time.time()
    reply = ''
    try:
        # This is likely be incorrect since the model will generate a raw output_ids
        reply = shared.model.generate(question, state)
        replies.append(reply)
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
    finally:
        t1 = time.time()
        original_tokens = len(encode(original_question)[0])
        new_tokens = len(encode(original_question + reply)[0]) - original_tokens
        print(f'Output generated in {(t1 - t0):.2f} seconds ({new_tokens / (t1 - t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
    return replies
