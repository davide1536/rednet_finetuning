import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import get_loaders, save_checkpoint, check_accuracy,CrossEntropyLoss2d
from rednet import load_rednet
import numpy as np
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_WORKERS = 4
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/training/rgb/"
TRAIN_DEPTH_DIR = "data/training/depth/"
TRAIN_MASK_DIR = "data/training/semantic/"
VAL_IMG_DIR = "data/validation/rgb/"
VAL_DEPTH_DIR = "data/validation/depth/"
VAL_MASK_DIR = "data/validation/semantic/"
def train_fn(loader, model, optimizer, loss_fn, scaler):
	loop = tqdm(loader)
	for batch_idx, (rgb_data,depth_data, targets) in enumerate(loop):
		
		rgb_data = rgb_data.permute(0, 2, 3, 1)
		#depth_data = depth_data.permute(0,3,1,2)
		rgb_data = rgb_data.to(device=DEVICE)
		depth_data = depth_data.to(device=DEVICE)
		targets = targets.to(device=DEVICE)
		with torch.cuda.amp.autocast():
			pred_values = model(rgb_data, depth_data)
			loss = loss_fn(pred_values, targets)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		loop.set_postfix(loss = loss.item())

	
	
def main():
	train_transform = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            ToTensorV2(),
        ],
    )

	ckpt_path = "/home/dfranzos/storage/SocialObjectNav/data/rednet_semmap_mp3d_tuned.pth"
	model = load_rednet(DEVICE, ckpt = ckpt_path, resize = True, stabilize = False)
	for param in model.rednet.parameters():
		param.requires_grad = False
	#model.rednet.forward_downsample.requires_grad = False
	#model.rednet.forward_upsample.requires_grad = True
	#model.rednet.final_deconv_custom.requires_grad = False
	for param in model.rednet.final_deconv_custom.parameters():
		param.requires_grad = True
	for param in model.rednet.final_conv.parameters():
		param.requires_grad = True
	for param in model.rednet.agant0.parameters():
		param.requires_grad = True
	for param in model.rednet.agant1.parameters():
		param.requires_grad = True
	for param in model.rednet.agant2.parameters():
		param.requires_grad = True
	for param in model.rednet.agant3.parameters():
		param.requires_grad = True
	for param in model.rednet.out2_conv_custom.parameters():
		param.requires_grad = True
	for param in model.rednet.out3_conv_custom.parameters():
		param.requires_grad = True
	for param in model.rednet.out4_conv_custom.parameters():
		param.requires_grad = True
	for param in model.rednet.out5_conv_custom.parameters():
		param.requires_grad = True
	for param in model.rednet.deconv4.parameters():
		param.requires_grad = True
	for param in model.rednet.deconv3.parameters():
		param.requires_grad = True
	for param in model.rednet.deconv2.parameters():
		param.requires_grad = True
	for param in model.rednet.deconv1.parameters():
		param.requires_grad = True
	
	CEL_weighted = CrossEntropyLoss2d()
	optimizer = torch.optim.SGD(model.rednet.parameters(), lr=LEARNING_RATE)
	train_loader, val_loader = get_loaders(
	TRAIN_IMG_DIR,
	TRAIN_DEPTH_DIR,
	TRAIN_MASK_DIR,
	VAL_IMG_DIR,
	VAL_DEPTH_DIR,
	VAL_MASK_DIR,
	BATCH_SIZE,
	train_transform,
	None,
	NUM_WORKERS,
	PIN_MEMORY,
	)
	check_accuracy(val_loader, model, device = DEVICE)
	scaler = torch.cuda.amp.GradScaler()
	for epoch in range(NUM_EPOCHS):
		train_fn(train_loader, model, optimizer, CEL_weighted, scaler)

		checkpoint = {
		"state_dict": model.state_dict(),
		"optimizer":optimizer.state_dict(),
        	}
		checkpoint_name = "ckpt" + str(epoch)
		save_checkpoint(checkpoint, checkpoint_name)

        	# check accuracy
		check_accuracy(val_loader, model, device=DEVICE)
if __name__ == "__main__":
	main()
