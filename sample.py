"""
Sample from a trained model

Usage:

With speculative sampling:
python sample.py --out_dir=out-shakespeare-char --device=mps --out_ckpt_path='ckpt-blsz128-bs64-nl4-nh4-nmbd256-iters5k-drop01.pt' \
    --use_speculative_sampling=True --draft_length=5 --draft_sample_ckpt='ckpt-blsz128-bs64-nl4-nh4-nmbd256-iters5k-drop01.pt'

With beam search:
python sample.py --out_dir=out-shakespeare-char --device=mps --out_ckpt_path='ckpt-blsz128-bs64-nl4-nh4-nmbd256-iters5k-drop01.pt' \
    --use_beam_search=True --num_beams=5

With top_p:
python sample.py --out_dir=out-shakespeare-char --device=mps --out_ckpt_path='ckpt-blsz128-bs64-nl4-nh4-nmbd256-iters5k-drop01.pt' \
    --top_p=0.95 --top_k=-1

With both top_p and beam search:
python sample.py --out_dir=out-shakespeare-char --device=mps --out_ckpt_path='ckpt-blsz128-bs64-nl4-nh4-nmbd256-iters5k-drop01.pt' \
    --use_beam_search=True --num_beams=5 --top_p=0.9 --max_new_tokens=30

With both top_p and speculative sampling:
python sample.py --out_dir=out-shakespeare-char --device=mps --out_ckpt_path='ckpt-blsz128-bs64-nl4-nh4-nmbd256-iters5k-drop01.pt' \
    --use_speculative_sampling=True --draft_length=5 --draft_ckpt_path='ckpt-blsz128-bs64-nl4-nh4-nmbd256-iters5k-drop01.pt' \
    --top_p=0.9
"""
import os
import pickle
from contextlib import nullcontext
import time

import torch
import tiktoken

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'  # ignored if init_from is not 'resume'
out_ckpt_path = 'ckpt.pt'
start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 5  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
top_p = .95  # .95  # .95
use_beam_search = False
num_beams = 5
use_speculative_sampling = False
draft_length = 5
draft_ckpt_path = 'ckpt-blsz128-bs64-nl2-nh4-nmbd128-iters5k-drop01.pt'
# blsz128-bs64-nl2-nh4-nmbd128-iters5k-drop01.pt'
seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available(
) and torch.cuda.is_bf16_supported(
) else 'float16'  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open(
    'configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
    device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, out_ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

# Speculative Sampling Model
if use_speculative_sampling:
    draft_checkpoint = torch.load(os.path.join(out_dir, draft_ckpt_path),
                                  map_location=device)
    draft_model = GPT(GPTConfig(**draft_checkpoint['model_args']))
    state_dict = draft_checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    draft_model.load_state_dict(state_dict)
    draft_model.eval()
    draft_model.to(device)
else:
    draft_model = None

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint[
        'config']:  # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'],
                             'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        times = []
        for k in range(num_samples):
            t = time.time()
            if use_beam_search:
                results = model.generate_with_beam_search(
                    x,
                    max_new_tokens,
                    num_beams=num_beams,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    return_beams=True,
                    return_log_probs=True)
                y_beams = results['beams']
                log_probs = results['log_probs']
                # shape of y_beams: [1, num_beams, max_new_tokens + 1]
                for num_beam in range(num_beams):
                    print('BEAM %d' % num_beam,
                          'log_prob: %.3f' % log_probs[0, num_beam])
                    print(decode(y_beams[0, num_beam].tolist()))
                print('---------------')
            elif use_speculative_sampling:
                y = model.generate_with_speculative_sampling(
                    draft_model,
                    draft_length,
                    x,
                    max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p)
                print(decode(y[0].tolist()))
                print('---------------')
            else:
                y = model.generate(x,
                                   max_new_tokens,
                                   temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p)
                # Shape of y: torch.Size([1, 501])
                print(decode(y[0].tolist()))
                print('---------------')
            times.append(time.time() - t)
        print('\n')
        print(
            'Average / Min / Max generation time per (%d) sample: %.3f / %.3f / %.3f'
            % (len(times), sum(times) / len(times), min(times), max(times)))
        print('\n')
