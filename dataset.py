import os
import numpy as np
import torch
import pickle
import math
from torch.utils.data import Dataset
class HabitatPersonDataset(Dataset):
	def __init__(self, image_dir, depth_dir, mask_dir, transform=None):
		self.image_dir = image_dir
		self.depth_dir = depth_dir
		self.mask_dir = mask_dir
		self.images = os.listdir(image_dir)
		self.transform = transform
	def __len__(self):
		return len(self.images)
	def __getitem__(self,index):
		#print("index ", index)
		image_path = os.path.join(self.image_dir, "data_"+str(index))
		depth_path = os.path.join(self.depth_dir, "data_"+str(index))
		mask_path = os.path.join(self.mask_dir, "data_"+str(index))
		f_rgb = open(image_path, "rb")
		f_depth = open(depth_path, "rb")
		f_mask = open(mask_path, "rb")
		image = pickle.load(f_rgb)
		depth = pickle.load(f_depth)
		mask = pickle.load(f_mask)
		f_rgb.close()
		f_depth.close()
		f_mask.close()
		#mask = torch.load(self.mask_dir)[index]
		#self.transform = None
		if self.transform is not None:
			augmentations = self.transform(image = image, depth = depth, mask = mask)
			image = augmentations["image"]
			depth = augmentations["depth"]
			mask = augmentations["mask"]
		return image,depth,mask
