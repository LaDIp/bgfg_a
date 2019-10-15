# bgfg_a
a collection of opencv **background segmentation** method with simple **denoise**, display functionalities are also provided.

## Prerequisite
~~~~
# basic
pip install opencv
# if you want to use bgmodels in cv2.bgsegm and cv2.cuda
pip install opencv-contrib-python 
~~~~

## Usage
~~~~
from bgfg_a import bgfg_a

# initialize background model
bgfg = bgfg_a(model, speed) 

# get foreground mask
fgmask = bgfg.apply(imput_image)

# display resulting image
# 1. original foreground mask
display_im = fgmask
# 2. color foreground mask
display_im = bgfg.mask_fusion(imput_image, fgmask)
# 3. foreground with bounding boxes and NMS
display_im = bgfg.foreground_bbox(display_im, fgmask, True)
cv2.imshow('new', display_im)
~~~~

you can try to run **bgfg_demo.py** to see all functionalities of bgfg_a and how they work.

## Author
terry0201@gmail.com
