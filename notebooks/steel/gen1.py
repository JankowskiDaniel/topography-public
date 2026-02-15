import numpy as np
from noise import pnoise2
import cv2
import matplotlib.pyplot as plt
import random
import os
import cv2
import numpy as np
import glob 
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
from tqdm import tqdm
sys.path.append('../..')

from src.data_generation.noise_controllers.noise_pizza import PizzaController

# Function to generate Perlin noise
def generate_perlin_noise(width, height, scale=100, octaves=6, persistence=0.7, lacunarity=2.0):
    # Random offset for x and y coordinates
    offset_x = random.uniform(0, 1000)
    offset_y = random.uniform(0, 1000)
    
    noise = np.zeros((height, width), dtype=np.float32)
    
    for i in range(height):
        for j in range(width):
            # Shift the Perlin noise grid with random offsets
            noise[i][j] = pnoise2((i + offset_x) / scale, (j + offset_y) / scale, 
                                  octaves=octaves, persistence=persistence, lacunarity=lacunarity)
    
    # Normalize to [0, 255] range
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255
    return noise.astype(np.uint8)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convert to grayscale if necessary
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced

controller = PizzaController(
    **{"nr_of_pizzas": (3,5),
    "center_point": (320, 240),
    "channels": 1,
    "strength": (20,30)}
)

GRID_SIZES = [(2, 2), (4, 4), (8, 8), (16, 16)]
CLIP_RANGE = [1.5, 2.2]
PERSISTANCE_RANGE = [0.7, 0.9]

path_to_save = "../../data/generated/perlin_ceramic_test/"
avg_path = "../../data/generated/perlin_ceramic_test/pure_average/"
avg_params = pd.read_csv(avg_path + "parameters.csv")
perlin_params = []

# take first 2500 rows
avg_params = avg_params.iloc[:5000]

counter = 0
alpha = 0.45
# iterate the dataframe
for index, row in tqdm(avg_params.iterrows()):
    # load the image
    image_path = avg_path + row["filename"]
    generated_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    grid_size = random.choice(GRID_SIZES)
    clip_limit = random.uniform(1.0, 2.0)
    persistance = random.uniform(0.7, 0.9)

    perlin_noise = generate_perlin_noise(generated_image.shape[1], generated_image.shape[0], scale=100, octaves=6, persistence=persistance, lacunarity=2.0)

    blended_image = cv2.addWeighted(generated_image, 1 - alpha, perlin_noise, alpha, 0)
    blended_image = apply_clahe(blended_image, tile_grid_size=grid_size, clip_limit=clip_limit)
    blended_image = controller.generate(blended_image, row["epsilon"])
    filename = f"perlin_{counter:05d}.png"
    cv2.imwrite(path_to_save + filename, blended_image)

    perlin_params.append({
        "width": generated_image.shape[1],
        "height": generated_image.shape[0],
        "epsilon": row["epsilon"],
        "ring_center_width": row["ring_center_width"],
        "ring_center_height": row["ring_center_height"],
        "min_brightness": row["min_brightness"],
        "max_brightness": row["max_brightness"],
        "filename": filename,
        "grid_size": grid_size,
        "clip_limit": clip_limit,
        "persistance": persistance
    })

    counter += 1

df = pd.DataFrame(perlin_params)
df.to_csv(path_to_save + "parameters.csv", index=False)
