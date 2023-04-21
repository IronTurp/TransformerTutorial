# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 23:31:54 2023

@author: jfturpin
"""

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def preprocess_image(image_path):
    transform = Compose([
        Resize(224),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)

def postprocess_feature_map(feature_map):
    feature_map = feature_map.squeeze().cpu().numpy()
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
    return (feature_map * 255).astype(np.uint8)

def overlay_feature_map_on_image(image, feature_map):
    overlay = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)
    overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

def main(image_path):
    # Load the DINO model
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model.eval()

    # Preprocess the input image
    input_tensor = preprocess_image(image_path)

    # Extract the feature map
    with torch.no_grad():
        feature_map = model(input_tensor)[0]

    # Postprocess the feature map
    feature_map = postprocess_feature_map(feature_map)

    # Load the original image using OpenCV
    original_image = cv2.imread(image_path)

    # Overlay the feature map on the original image
    result = overlay_feature_map_on_image(original_image, feature_map)

    # Display the result
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r"img_path"
    main(image_path)

