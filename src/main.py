# Written by Harish Prakash for Tiger Analytics on Dec 24, 2025. 

"""
Class to implement image resolution enhancement model. 
"""
import os
import math
import numpy as np
import rasterio
from PIL import Image
import tensorflow
from tensorflow.keras.models import load_model
from pathlib import Path
import logging
from datetime import datetime

class ImageResEnhancer:
    def __init__(self, modelPath: str | Path) -> None:
        '''
        Default method to implement __init__ initializer. 
        
        Args: 
            modelPath (Path) : Path to the model file. 

        Returns: 
            None 
        '''
        self.modelPath = modelPath
        self.generator = load_model(self.modelPath, compile=False)

    def patchify_input(self, image_array: np.ndarray) -> np.ndarray:
        '''
        Method to split the input image array into patches.  
        
        Args: 
            image_array (np.ndarray) : A (b, 1024, 1024) array captured by the satellite. b = bands.   

        Returns: 
            patches (np.ndarray) :  A (k, k, p, p, b) array where k = h // patchsize or 
                                    w // patchsize, p = patchsize, b = bands.
        '''
        img = np.moveaxis(image_array, 0, -1)
        h, w, b = img.shape
        patch_size = 128 # hardcoded input dimension for our custom model. 

        assert h % patch_size == 0 and w % patch_size == 0, "Image dimensions must be multiples of 128"
        patches = img.reshape(h // patch_size, patch_size, w // patch_size, patch_size, b)
        patches = patches.transpose(0, 2, 1, 3, 4) 
        return patches 

    def unpatchify_result(self, image_stack: np.ndarray) -> np.ndarray:
        '''
        Method to collate the patches into a single array.  
        
        Args: 
            image_stack (np.ndarray): A (n, 1, 256, 256, 3)-dim array. n = total number of patches. 

        Returns: 
            unpatched (np.ndarray): A (256, 256, 3)-dim output array with the collated patches.  
        '''        
        stack = image_stack.squeeze(axis=1) 
        num_patches = stack.shape[0]
        grid_size = int(math.sqrt(num_patches))
        patch_h, patch_w, b = stack.shape[1:]
        
        reshaped = stack.reshape(grid_size, grid_size, patch_h, patch_w, b)
        transposed = reshaped.transpose(0, 2, 1, 3, 4)
        unpatched = transposed.reshape(grid_size * patch_h, grid_size * patch_w, b)
        return unpatched

    def modelinfer(self, image_array: np.ndarray) -> np.ndarray:
        '''
        Method to run model inference. 

        Args: 
            image_array (np.ndarray): A (128, 128, 3)-dim input to the model. 

        Returns: 
            genimg (np.ndarray): A (256, 256, 3)-dim output from the model. 
        '''
        genimg = self.generator.predict(image_array, verbose=0)
        return genimg

    def imgsave(self, outputpath: str | Path, genimg: np.ndarray, _input_name: str) -> None:
        '''
        Method to save output in .tif format.  

        Args: 
            outputpath (Path): Path to store the output .tif file. 
            genimg (np.ndarray): A (256, 256, 3)-dim array. 
            _input_name (str): The name for saving output file. 

        Returns: 
            None
        '''
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

def main(modelpath: str | Path, inputpath: str | Path, outputpath: str | Path):
    '''
    Main function for file execution. 

    Args: 
        modelpath (Path): Path to the model file. 
        inputpath (Path): Path to the input .tif file. 
        outputpath (Path): Path to the output .tif file. 
    
    Returns: 
        None
    '''
    log_file = os.path.join(outputpath, "error_log.txt")
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    imgEnH = ImageResEnhancer(modelpath)
    for root, _, files in os.walk(inputpath):
        for _input_name in files:
            if _input_name.lower().endswith((".tiff", ".tif")):

                input_image_path = os.path.join(root, _input_name)
                with rasterio.open(input_image_path) as src:
                    image_array = src.read() 
                    patches = imgEnH.patchify_input(image_array)
                    _, _, p_h, p_w, p_b = patches.shape
                    
                    flat_patches = patches.reshape(-1, p_h, p_w, p_b)
                    _to_unpatch = []
                    print("Running Inference ...")
                    for patch in flat_patches:
                        patch_input = np.expand_dims(patch[:, :, :3], axis=0)
                        genimg = imgEnH.modelinfer(patch_input)
                        _to_unpatch.append(genimg)
                    
                    image_stack = np.stack(_to_unpatch, axis=0)
                    unpatched_image = imgEnH.unpatchify_result(image_stack)
                    imgEnH.imgsave(outputpath, unpatched_image, _input_name)

if __name__ == "__main__": 

    modelpath = "models/gen_e_50.h5"  
    inputpath  = "opt/ilc_player/data"
    outputpath = "opt/ilc_player/results"
    main(modelpath, inputpath, outputpath)