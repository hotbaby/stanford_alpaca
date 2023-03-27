# encoding: utf8

import os
import json
import time
import torch
from transformers import LlamaTokenizer
from transformers import LlamaConfig
from transformers import LlamaForCausalLM
from transformers import LlamaModel
from transformers import Trainer


model_dir = "/data/models/alpaca"
config_path = os.path.join(model_dir, "config.json")
tokenizer_path = os.path.join(model_dir, "tokenizer.model")



print("Loading tokenizer model.")
tokenizer = LlamaTokenizer(tokenizer_path)

prompt = "海天瑞声"
enc_tokens = tokenizer.encode(prompt)
enc_tokens = torch.tensor(tokenizer.encode("海天瑞声"), dtype=torch.long, device=torch.device("cuda:0")).unsqueeze(0)
print(f"prmpt: {prompt}, tokens: {enc_tokens}")


with open(config_path) as f:
    config_json = json.load(f)

print("Loading Alpaca LLM")
# llama_config = LlamaConfig(**config_json)
model = LlamaForCausalLM.from_pretrained(model_dir)
model = model.cuda()

print("inference ...")
prompts = [
    "I believe the meaning of life is",
    "介绍下百度公司，"
]

for prompt in prompts:
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=torch.device("cuda:0")).unsqueeze(0)
    result = model.generate(tokens, max_length=256)
    print(tokenizer.batch_decode(result, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

time.sleep(100)