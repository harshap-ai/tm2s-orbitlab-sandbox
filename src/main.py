from tensorflow.keras.models import load_model
from numpy.random import randint
import numpy as np
from PIL import Image
import os
import rasterio 
import patchify 
import math 

class ImageResEnhancer:
  def __init__(self, modelPath):
    self.modelPath = modelPath

  def loadfile(self,filepath):
    img = np.load(filepath)
    if img.shape[0] > 1:
      img = img[:1]
    return img

  def modelinfer(self,image_array):
    generator = load_model(self.modelPath, compile=False)
    src_image = image_array
    genimg = generator.predict(src_image)
    return src_image,genimg
  
  def patchify_input(self, image_array):

    numpy_image = np.moveaxis(image_array, 0, -1)
    assert numpy_image.shape[0] == numpy_image.shape[1], "Please match height and width dimensions"
    patches = patchify.patchify(numpy_image, (128, 128, numpy_image.shape[2]), step=128)
    return patches 
  
  def unpatchify_result(self, image_stack):

    divisor = int(math.sqrt(image_stack.shape[0]))
    image_stack_squeezed = image_stack.squeeze(axis=1) 
    reshaped_image_stack = image_stack_squeezed.reshape(divisor, divisor, 1, image_stack_squeezed.shape[1], image_stack_squeezed.shape[2], image_stack_squeezed.shape[3])
    #reshape_image_stack.shape = (divisor, divisor, 1, 256, 256, 3)

    target_h = reshaped_image_stack.shape[0] * reshaped_image_stack.shape[3]
    target_w = reshaped_image_stack.shape[1] * reshaped_image_stack.shape[4]
    target_c = reshaped_image_stack.shape[5]
    unpatched_image = patchify.unpatchify(reshaped_image_stack, (target_h, target_w, target_c))
    return unpatched_image
    
if __name__ == "__main__": 
    
    modelpath = f"/workspace/models/gen_e_50.h5"
    inputpath  = f"/opt/ilc_player/data"
    outputpath = f"/opt/ilc_player/results"

    imgEnH = ImageResEnhancer(modelpath)

    for _input_name in os.listdir(inputpath):
      if _input_name.endswith(".tiff", ".tif"):
          input_image = os.path.join(inputpath, _input_name)
          with rasterio.open(input_image) as input:
            # Assuming array with dimension order (Bands, Height, Width)
            image_array = input.read()
            patches = imgEnH.patchify_input(image_array)
            
            flat_patches = patches.reshape(-1, 128, 128, 4)
            _to_unpatch = list()
            for _input in flat_patches:
              # Assuming that we take the first 3 channels for our model.  
              _, genimg = imgEnH.modelinfer(np.expand_dims(_input[:, :, :3], axis=0))
              _to_unpatch.append(genimg)
            image_stack = np.stack(_to_unpatch, axis=0)
            unpatched_image = imgEnH.unpatchify_result(image_stack)

            unpatched_image.save(os.path.join(outputpath, f"{os.path.splittext(_input_name)[0]}.png"))
           
      else: 
         raise ValueError("Incorrect image format")
      
      src_image, genimg = imgEnH.modelinfer(  )
      imgEnH.imgsave(outputpath,genimg)
