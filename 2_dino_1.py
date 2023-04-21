# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 21:51:09 2023

@author: jfturpin
"""

import cv2
import torch
import mss
import numpy as np
from torchvision import transforms

# Load the DINO model
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
model.eval()

# Initialize the screen capture object
with mss.mss() as sct:
    # Define the region of the screen to capture
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

    # Initialize the OpenCV window
    cv2.namedWindow('Dino', cv2.WINDOW_NORMAL)

    # Define the image transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    while True:
        # Capture the screen
        img = np.array(sct.grab(monitor))

        # Preprocess the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = transform(img_rgb).unsqueeze(0)

        # Pass the image through the DINO model
        with torch.no_grad():
            features = model.get_last_selfattention(img_t)

        # Extract and normalize the attention map
        attention_map = features.mean(dim=1).squeeze().numpy()
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        # Resize the attention map to match the input image size
        attention_map_resized = cv2.resize(attention_map, (1920, 1080))

        # Apply a colormap to the attention map
        colored_attention_map = cv2.applyColorMap(np.uint8(255 * attention_map_resized), cv2.COLORMAP_JET)

        # Convert the attention map to a 3-channel image
        colored_attention_map_3ch = cv2.cvtColor(colored_attention_map, cv2.COLOR_BGR2RGB)

        # Overlay the attention map onto the original image (ignoring the alpha channel)
        overlay = cv2.addWeighted(img[:, :, :3], 0.5, colored_attention_map_3ch[:, :, :3], 0.5, 0)


        # Display the output in the OpenCV window
        cv2.imshow('Dino', overlay)

        # Exit the script when the 'Q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the OpenCV window
    cv2.destroyAllWindows()
