# Written by Harish Prakash for Tiger Analytics on Dec 24, 2025. 

import os
import math
import numpy as np
import rasterio
from PIL import Image
from tensorflow.keras.models import load_model

class ImageResEnhancer:
    def __init__(self, modelPath):
        self.modelPath = modelPath
        self.generator = load_model(self.modelPath, compile=False)

    def patchify_input(self, image_array):
        # image_array: (Bands, H, W) -> (H, W, Bands)
        img = np.moveaxis(image_array, 0, -1)
        h, w, c = img.shape
        patch_size = 128 # as taken by our custom model. 

        assert h % patch_size == 0 and w % patch_size == 0, "Image dimensions must be multiples of 128"
        patches = img.reshape(h // patch_size, patch_size, w // patch_size, patch_size, c)
        patches = patches.transpose(0, 2, 1, 3, 4) 
        return patches 

    def unpatchify_result(self, image_stack):
        # image_stack: (Total_Patches, 1, 256, 256, 3) 
        stack = image_stack.squeeze(axis=1) # Remove the extra dimension
        num_patches = stack.shape[0]
        grid_size = int(math.sqrt(num_patches))
        patch_h, patch_w, c = stack.shape[1:]
        
        reshaped = stack.reshape(grid_size, grid_size, patch_h, patch_w, c)
        transposed = reshaped.transpose(0, 2, 1, 3, 4)
        unpatched = transposed.reshape(grid_size * patch_h, grid_size * patch_w, c)
        return unpatched

    def modelinfer(self, image_array):
        # image_array shape expected: (1, 128, 128, 3)
        genimg = self.generator.predict(image_array, verbose=0)
        return genimg

    def imgsave(self, outputpath, genimg, _input_name):
        clean_name = os.path.splitext(_input_name)[0]
        save_path = os.path.join(outputpath, f"{clean_name}.tif")
        
        if genimg.ndim == 4:
            genimg = np.squeeze(genimg, axis=0)

        output_data = np.moveaxis(genimg, -1, 0)
        output_data = np.clip(output_data, 0, 255).astype(np.uint8)
        
        count, height, width = output_data.shape

        with rasterio.open(
            save_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=count,
            dtype='uint8', 
            nodata=0       
        ) as dst:
            dst.write(output_data)

if __name__ == "__main__": 
    modelpath = "/models/gen_e_50.h5"  
    inputpath  = "/opt/ilc_player/data"
    outputpath = "/opt/ilc_player/results"

    imgEnH = ImageResEnhancer(modelpath)

    for root, dirs, files in os.walk(inputpath):
        for _input_name in files:
            if _input_name.lower().endswith((".tiff", ".tif")):
                input_image_path = os.path.join(root, _input_name)

                with rasterio.open(input_image_path) as src:
                    image_array = src.read() 
                    patches = imgEnH.patchify_input(image_array)
                    _, _, p_h, p_w, p_c = patches.shape
                    
                    _to_unpatch = []
                    flat_patches = patches.reshape(-1, p_h, p_w, p_c)
                    
                    for patch in flat_patches:
                        # Taking first 3 channels and adding batch dimension
                        patch_input = np.expand_dims(patch[:, :, :3], axis=0)
                        genimg = imgEnH.modelinfer(patch_input)
                        _to_unpatch.append(genimg)
                    
                    # Reconstruct
                    image_stack = np.stack(_to_unpatch, axis=0)
                    unpatched_image = imgEnH.unpatchify_result(image_stack)
                    imgEnH.imgsave(outputpath, unpatched_image, _input_name)

