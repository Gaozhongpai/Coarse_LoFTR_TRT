#%%
import os
import sys
import cv2
import numpy as np
import torch
from loftr import LoFTR
from loftr.utils.cvpr_ds_config import default_cfg
from utils import make_query_image, get_coarse_match, make_student_config

repo = './'
device = 'cuda'

model_cfg = make_student_config(default_cfg)

matcher = LoFTR(config=model_cfg)
checkpoint = torch.load(os.path.join(repo, 'weights', 'LoFTR_teacher.pt'))
if checkpoint is not None:
  state_dict = checkpoint['model_state_dict']
  matcher.load_state_dict(state_dict, strict=False)
  device = torch.device(device)
  matcher = matcher.eval().to(device=device)
  print('Successfully loaded pre-trained weights.')
else:
  print('Failed to load weights')
  
#%%
img_size = (model_cfg['input_width'], model_cfg['input_height'])
loftr_coarse_resolution = model_cfg['resolution'][0]

img0_orig = cv2.imread('out320_undistorted.png')
img1_orig = cv2.imread('out325_undistorted.png')
shape0 = img0_orig.shape[:2]  # current shape [height, width]
shape1 = img1_orig.shape[:2]  # current shape [height, width]
ratio = min(img_size[1] / shape0[0], img_size[0] / shape0[1], 
        img_size[1] / shape1[0], img_size[0] / shape1[1])
img0_orig, (dw0, dh0) = make_query_image(img0_orig, ratio)
img1_orig, (dw1, dh1) = make_query_image(img1_orig, ratio)

img0 = torch.from_numpy(img0_orig)[None][None].to(device=device) / 255.0
img1 = torch.from_numpy(img1_orig)[None][None].to(device=device) / 255.0

f = os.path.join(repo, 'weights', 'LoFTR.onnx')
input_names = ['image0', "image1"]
output_names = ["conf_matrix", "sim_matrix"]
torch.onnx.export(matcher, (img0, img1), f, 
                          verbose=False, 
                          opset_version=12, 
                          input_names=input_names,
                          output_names=output_names)

#%%
# Find matches
with torch.no_grad():
  conf_matrix, _ = matcher(img0, img1)
  conf_matrix = conf_matrix.cpu().numpy()

  mkpts0, mkpts1, mconf = get_coarse_match(conf_matrix, img_size[1], img_size[0], loftr_coarse_resolution)

  # filter only the most confident features
  n_top = 100
  indices = np.argsort(mconf)[::-1]
  indices = indices[:n_top]
  mkpts0 = mkpts0[indices, :]
  mkpts1 = mkpts1[indices, :]
  
#%%
# Show result
# %matplotlib inline
import matplotlib.pyplot as plt

def draw_features(image, features, img_size, color, draw_text=True):
  indices = range(len(features))
  sx = image.shape[1] / img_size[0]
  sy = image.shape[0] / img_size[1]

  for i, point in zip(indices, features):
    point_int = (int(round(point[0] * sx)), int(round(point[1] * sy)))
    cv2.circle(image, point_int, 2, color, -1, lineType=16)
    if draw_text:
      cv2.putText(image, str(i), point_int, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

draw_features(img0_orig, mkpts0, img_size, color=(0, 255, 0))
draw_features(img1_orig, mkpts1, img_size, color=(0, 255, 0))

# combine images
res_img = np.hstack((img0_orig, img1_orig))
res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize = (20, 10))
plt.imshow(res_img)
plt.show()
# %%
