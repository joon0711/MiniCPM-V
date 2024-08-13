import os
import json
import random

#data_path = "/home/ubuntu/joon/minicpm/MiniCPM-V/finetune/data/pdf2markdown_train_snapshots/2024-08-12"
data_path = "<Location to Datasets>"
image_list = os.listdir(f"{data_path}/images")
markdown_list = os.listdir(f"{data_path}/markdown")

image_list.sort()
markdown_list.sort()

'''
[
    {
      "id": "0",
      "image": "/home/ubuntu/joon/minicpm/MiniCPM-V/bag_image.jpg",
      "conversations": [
            {
              "role": "user", 
              "content": "<image>\nHow many desserts are on the white plate?"
            }, 
            {
                "role": "assistant", 
                "content": "There are three desserts on the white plate."
            }
        ]
    }
]

'''
dataset = []
idx = 0

for image_file in image_list:
    example = {}
    file_name = image_file.split(".")[0]
    image_path = os.path.join(data_path, "images", image_file)
    markdown_path = os.path.join(data_path, "markdown", file_name+".md")
    
    if not os.path.exists(markdown_path):
        print(f"skipping: Could not find {markdown_path}")
        continue
    with open(markdown_path) as f:
        markdown_content = f.read()
    
    conversations = []
    question = "<image>\nCan you convert the information in the image to a markdown file. Only output the markdown, nothing else"
    conversations.append({
        "role": "user",
        "content": question,
    })
    conversations.append({
        "role": "assistant",
        "content": markdown_content,
    })

    example["id"] = str(idx)
    example["image"] = image_path
    example["conversations"] = conversations
    dataset.append(example)
    
    idx += 1

random.shuffle(dataset)

train_dataset = dataset[:-10]
eval_dataset = dataset[-10:]

with open("dataset_train.json", "w") as f:
    f.write(json.dumps(train_dataset, indent=2))
    
with open("dataset_eval.json", "w") as f:
    f.write(json.dumps(eval_dataset, indent=2))

print(f"Training Data: {len(train_dataset)}")
print(f"Eval Data: {len(eval_dataset)}")
