import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import os
import tqdm
import re
import json

from fsspec import open
import fsspec
import io
import torch.distributed as dist
from fsspec.utils import other_paths

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

def download(remote_path: str, local_path: str):
    """Downloads all files from a remote directory to a local directory.

    Assumes dist is initialized.

    Args:
        remote_path (str): The path to the remote directory (tested for gcs).
        local_path (str): The path to the local directory.
    """
    remote_fs, _, _ = fsspec.get_fs_token_paths(remote_path)

    # get list of all files in remote directory
    rpaths = remote_fs.expand_path(remote_path, recursive=True)
    rpaths = [rpath for rpath in rpaths if remote_fs.isfile(rpath)]

    # make corresponding file paths in local directory
    lpaths = other_paths(rpaths, local_path)

    for rpath, lpath in zip(rpaths, lpaths, strict=True):
        remote_fs.get_file(rpath, lpath)


def generate():
    #model_path = "openbmb/MiniCPM-V-2_6"
    '''
    model_path = "gs://cohere-dev-central-2/joon/vision/minicpmv26"
    inputs = ["gs://cohere-dev-central-2/joon/vision/pdf2markdown_train_snapshots/2024-08-13/images/000014-001.png"]
    output = "gs://cohere-dev-central-2/joon/vision/minicpm_gen"
    '''

    model_path = os.environ.get("MODEL_PATH")
    inputs = os.environ.get("INPUTS")
    output = os.environ.get("OUTPUT_DIR")

    with fsspec.open(inputs, "r") as f:
        inputs = json.load(f)


    params = os.environ.get("PARAMS")
    params = json.loads(params)
    
    #Sampling
    """
    params = {
                'sampling': False,
                'num_beams': 3,
                'repetition_penalty': 1.2,
                "max_new_tokens": 6000,
                "max_inp_length": 4352
            }

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

    # Download model to local
    local_path = "minicpmv26"
    print(f"Download started {model_path}")
    download(model_path, local_path)

    model = AutoModel.from_pretrained(local_path, trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

    model.processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)

    for example in tqdm.tqdm(inputs):
        image_path = example["image"]
        image = read_image(image_path)
        #question = '\nCan you convert the information in the image to a markdown file. Only output the markdown, nothing else'
        question = example['conversations'][0]['content'].split("<image>")[-1]
        msgs = [{'role': 'user', 'content': [image, question]}]

        res = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            **params
        )

        res = clean(res)
        print(res)

        md_path = os.path.join(output, os.path.basename(image_path)[0:-4]+".md")
        with open(md_path, "w") as fOut:
            fOut.write(res)


if __name__ == "__main__":
    generate()