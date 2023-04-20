# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:55:55 2023

@author: jeanfrancois.turpin
"""

import torch
#import cv2
from PIL import Image
#import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

print(f'torch: {torch.__version__}')
print(f'cuda enable: {torch.cuda.is_available()}')
print(f'current_device: {torch.cuda.current_device()}')
print(f'device: {torch.cuda.device(0)}')
print(f'device_count: {torch.cuda.device_count()}')
print(f'get_device_name: {torch.cuda.get_device_name(0)}')
print(torch.__version__)

#%% Loading the model
model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

#%%  Download the classes, easier to understand the output

# Download ImageNet category names for nicer display
classes_url = r"https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

response = requests.get(classes_url)  # download the txt file

if response.status_code == 200:
    # the request was successful, proceed to open the file
    txt_content = response.content.decode('utf-8')  # decode the content to text
    with open("imagenet_classes.txt", "w") as f:
        f.write(txt_content)

        # read the contents of the file and strip any whitespace
        imagenet_categories = [s.strip() for s in txt_content.split("\n")]

        # print the first 10 categories
        print(imagenet_categories[:10])
else:
    # the request was not successful
    print(f"Failed to download file. Status code: {response.status_code}")
    
#%% Load the model

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256, Image.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

#%% Test it on an image of a wolf, works, Doberman with tail and uncropped ear and small size, fails
img = Image.open(requests.get("https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png", stream=True).raw)
#img = Image.open(requests.get("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSt9977XsUQdrtsSzVSEjTz2m0NqQGKvgFbkJqS6_6jJA&s", stream=True).raw)
img = transform(img)[None,]
out = model(img)

# and convert them into probabilities
scores = torch.nn.functional.softmax(out, dim=-1)[0]

# finally get the index of the prediction with highest score
topk_scores, topk_label = torch.topk(scores, k=5, dim=-1)

for i in range(5):
  pred_name = imagenet_categories[topk_label[i]]
  print(f"Prediction index {i}: {pred_name:<25}, score: {topk_scores[i].item():.3f}")

clsidx = torch.argmax(out)
print("Class number: {0}".format(clsidx.item()))
