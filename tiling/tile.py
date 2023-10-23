from PIL import Image
import os
import sys
import numpy as np

def tile_image(image_path, H, W, output_directory, fill_color=(0, 0, 0)):
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception as e:
        print(f"An error occurred while opening the image: {e}", file=sys.stderr)
        sys.exit(1)

    image_width, image_height = image.size

    # Calculate the dimensions of the grid
    columns = -(-image_width // W)  # Ceiling division
    rows = -(-image_height // H)    # Ceiling division

    # Create a new blank image with the extended size, filled with the specified color
    extended_width = columns * W
    extended_height = rows * H
    extended_image = Image.new(image.mode, (extended_width, extended_height), fill_color)
    extended_image.paste(image)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Cut the image into tiles
    tiles = []
    for i in range(rows):
        for j in range(columns):
            left = j * W
            upper = i * H
            right = left + W
            lower = upper + H

            tile = extended_image.crop((left, upper, right, lower))
            tile_path = f"{output_directory}/tile_{i}_{j}.png"
            tile.save(tile_path)
            tiles.append(tile_path)

    return tiles

# Example usage
tile_image("./totile.png", 64, 64, "./tiled", fill_color=(255, 255, 255))

