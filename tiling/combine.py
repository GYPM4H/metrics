from PIL import Image
import os
import sys

def combine_tiles(input_directory, H, W, output_image_path):
    # List all files in the directory and read points of tiles and their names
    tiles = []
    for filename in os.listdir(input_directory):
        if filename.startswith("tile_") and filename.endswith(".png"):
            row_col = filename.replace("tile_", "").replace(".png", "").split("_")
            point = (int(row_col[1]) * W, int(row_col[0]) * H)  # (x, y)
            tiles.append((point, os.path.join(input_directory, filename)))

    if not tiles:
        print("No tiles were found in the input directory.", file=sys.stderr)
        sys.exit(1)

    # Find the maximum x and y coordinates to calculate the size of the combined image
    max_x = max(tile[0][0] for tile in tiles) + W
    max_y = max(tile[0][1] for tile in tiles) + H

    # Create a new image of the size we calculated
    combined_image = Image.new("RGB", (max_x, max_y))

    # Paste the tiles into the image
    for point, filepath in tiles:
        tile = Image.open(filepath)
        combined_image.paste(tile, point)

    # Save the new image
    combined_image.save(output_image_path)

# Example usage
combine_tiles("./tiled", 64, 64, "./untiled.png")
