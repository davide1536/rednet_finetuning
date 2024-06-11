import torch
from torch import nn
import torchvision
from dataset import HabitatPersonDataset
from torch.utils.data import DataLoader
import numpy as np
class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss2d, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            inputs = inputs.unsqueeze(0)
            if not -1 in targets:
                mask = targets > 0
                targets_m = targets.clone()
                targets_m[mask] -= 1
                targets_m = targets_m.squeeze()
                targets_m = targets_m.unsqueeze(0).long()
            #print("targets shape", targets_m.shape)
            #print("inputs shape", inputs.shape)
            #print(inputs)
                inputs = inputs.float()
                loss_all = self.ce_loss(inputs, targets_m)
                losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))  
            
                
        total_loss = sum(losses)
        #print("loss grad_fn ", total_loss.grad_fn)
        return total_loss
def get_loaders(
	train_image_dir,
	train_depth_dir,
	train_mask_dir,
	val_image_dir,
	val_depth_dir,
	val_mask_dir,
	batch_size,
	train_transform,
	val_transform,
	num_workers = 1,
	pin_memory = True
):
	train_ds = HabitatPersonDataset(
		image_dir = train_image_dir,
		depth_dir = train_depth_dir,
		mask_dir = train_mask_dir,
		transform = train_transform,
	)
	train_loader = DataLoader(
		train_ds,
		batch_size = batch_size,
		num_workers = num_workers,
		pin_memory = pin_memory,
		shuffle = True,
	)
	val_ds = HabitatPersonDataset(
		image_dir = val_image_dir,
		depth_dir = val_depth_dir,
		mask_dir = val_mask_dir,
		transform = val_transform,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size = batch_size,
		num_workers = num_workers,
		pin_memory = pin_memory,
		shuffle = False,
	)
	return train_loader, val_loader
def check_accuracy(loader, model, device="cuda"):
	print("evaluation")
	i = 0
	num_correct = 0
	num_pixels = 0
	dice_score = 0
	num_correct_people = 0
	num_pixels_people = 0
	num_correct_no_people = 0
	num_pixels_no_people = 0
	model.eval()
	with torch.no_grad():
		for rgb_b,depth_b,targets_b in loader:
		
			if i%50 == 0:
				print(i)
			for rgb, depth, targets in zip(rgb_b, depth_b, targets_b):
				mask = targets > 0
				targets_m = targets.clone()
				targets_m[mask] -= 1
				rgb = rgb.to(device)
				depth = depth.to(device)
				rgb = torch.unsqueeze(rgb, 0)
				depth = torch.unsqueeze(depth,0)


				targets_m = targets_m.to(device)
				preds = torch.max(model(rgb, depth),1)[1]
				#print("target size ", targets.shape)
				#print("preds size ", preds.shape)
				#print("mask size ", mask.shape)
				preds = torch.squeeze(preds)
				targets_m = torch.squeeze(targets_m)
				targets = torch.squeeze(targets)
				no_people_index = targets != 22
				num_correct_no_people += (preds[no_people_index] == targets_m[no_people_index]).sum()
				num_pixels_no_people += torch.count_nonzero(targets!=22)
				num_correct += (preds == targets_m).sum()
				num_pixels += torch.numel(preds)
					
				if 22 in targets:
					people_index = targets == 22
					num_correct_people += (preds[people_index] == targets_m[people_index]).sum()
					num_pixels_people += torch.count_nonzero(targets==22)
			i += 1

	print("no people accuracy")
	print(num_correct_no_people/num_pixels_no_people)
	print("accuracy")
	print(num_correct/num_pixels*100)
	print("people accuracy")
	if num_pixels_people == 0:
		print("not people in batch")
	else:
		print(num_correct_people/num_pixels_people*100)
	model.train()


def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
