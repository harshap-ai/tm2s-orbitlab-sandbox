import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def display_tif(image_path: str | Path):
    """
    Read a .tif file and display it using matplotlib.
    
    Args:
        image_path (str | Path): Path to the .tif image file.
    """
    with rasterio.open(image_path) as src:

        image_data = src.read([1, 2, 3])
        image_data = np.moveaxis(image_data, 0, -1)
        # Normalize data to [0, 1] if it's 16-bit or high range
        if image_data.max() > 1.0:
            image_data = image_data / image_data.max()

        plt.figure(figsize=(10, 10))
        plt.imshow(image_data)
        plt.title(f"Displaying: {Path(image_path).name}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    outputpath = "opt/ilc_player/results"
    filepath = os.path.join(outputpath, "global_monthly_2017_08_mosaic_L15-1670E-1159N_6681_3552_13.tif")
    display_tif(filepath)