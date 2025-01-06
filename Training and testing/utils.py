import math
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import cv2
from skimage.metrics import structural_similarity as ssim

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)


def initialize_weights(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean):

    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    # PSNR += ssim(Iclean, Img, channel_axis=2, data_range=255)
    for i in range(Img.shape[0]):
        Iclean_1 = np.squeeze(Iclean[i,:,:,:]).transpose(1, 2, 0)
        Img_1 = np.squeeze(Img[i,:,:,:]).transpose(1, 2, 0)

        SSIM += ssim(Iclean_1, Img_1, channel_axis=-1, data_range=1)
    return (SSIM/Img.shape[0])

    
def data_augmentation(image, mode):
    # out = np.transpose(image, (1,2,0))
    out = image
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = torch.flip(out, dims=[2,3])
    elif mode == 2:
        # rotate counterwise 90 degree
        out = torch.rot90(out, dims=[2,3])
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = torch.rot90(out, dims=[2,3])
        out = torch.flip(out, dims=[2,3])
    elif mode == 4:
        # rotate 180 degree
        out = torch.rot90(out, k=2, dims=[2,3])
    elif mode == 5:
        # rotate 180 degree and flip
        out = torch.rot90(out, k=2, dims=[2,3])
        out = torch.flip(out, dims=[2,3])
    elif mode == 6:
        # rotate 270 degree
        out = torch.rot90(out, k=3, dims=[2,3])
    elif mode == 7:
        # rotate 270 degree and flip
        out = torch.rot90(out, k=3, dims=[2,3])
        out = torch.flip(out, dims=[2,3])
    return out


def variable_to_cv2_image(varim):
	r"""Converts a torch.autograd.Variable to an OpenCV image

	Args:
		varim: a torch.autograd.Variable
	"""
	nchannels = varim.size()[1]
	if nchannels == 1:
		res = (varim.data.cpu().numpy()[0, 0, :]*255.).clip(0, 255).astype(np.uint8)
	elif nchannels == 3:
		res = varim.data.cpu().numpy()[0]
		res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
		res = (res*255.).clip(0, 255).astype(np.uint8)
	else:
		raise Exception('Number of color channels not supported')
	return res


def normalize(data):
    return data/255