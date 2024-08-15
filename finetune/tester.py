import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import argparse
import os
import tqdm
import re 

from fsspec import open
import fsspec
import io

def clean(text):
    text = text.replace("```markdown", "").replace("```", "")

    # Replace images with their alt text
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'', text)

    # Remove links, keeping only the text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    return text

def read_image(image_path):
    with fsspec.open(image_path, "rb") as f:
        fsspec.asyn.iothread[0] = None
        fsspec.asyn.loop[0] = None
        buf = f.read()
        f_io = io.BytesIO(buf)
        image = Image.open(f_io).convert("RGB")
    return image 

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", nargs="+", help="input img file paths")
parser.add_argument("--output", help="output folder")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

model_path = "openbmb/MiniCPM-V-2_6"
#model_path = "output/output_minicpmv26_epoch_3"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model.processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)


params = {
            'sampling': False,
            'num_beams': 3,
            'repetition_penalty': 1.2,
            "max_new_tokens": 6000,
            "max_inp_length": 4352
        }


#Sampling
"""
params = {
            'sampling': True,
            'top_p': 0.8,
            'top_k': 100,
            'temperature':  0.7,
            'repetition_penalty': 1.05,
            "max_new_tokens": 2048,
            "max_inp_length": 4352
        }


params = {
            'sampling': True,
            'temperature':  0.1,
        }
"""

#Greedy decoding
"""
params = {
            'sampling': False,
        }
"""

for in_path in tqdm.tqdm(args.inputs):
    image = read_image(in_path)
    #image = Image.open(in_path).convert('RGB')
    question = '\nCan you convert the information in the image to a markdown file. Only output the markdown, nothing else'
    msgs = [{'role': 'user', 'content': [image, question]}]

    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        **params
    )

    res = clean(res)
    print(res)

    md_path = os.path.join(args.output, os.path.basename(in_path)[0:-4]+".md")
    with open(md_path, "w") as fOut:
        fOut.write(res)
    
