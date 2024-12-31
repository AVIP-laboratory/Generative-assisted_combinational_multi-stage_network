# Generative tandem neural network
Noise degrades image quality and can result in the loss of important information, making its removal or minimization essential. However, as noise levels increase, eliminating it becomes exponentially more challenging. This project proposes a **Generative tandem neural network (GTNN)** capable of restoring the textural details and patterns of original images lost due to extreme noise.

## Requirements
- pytorch 1.21
- Python 3.9
- torchvision
- scikit-image
- os
- numpy

## Dataset description 
The relationship between the noisy image 𝑞, noise 𝑛, and high-quality image 𝑐 is defined as 𝑞=𝑛+𝑐. The noise 𝑛 is modeled as Gaussian noise, represented by $n \sim \mathcal{N}(\mu, \sigma^2)$, where $\sigma$ is the standard deviation. For training, the noise is randomly smapled within the range [10, 110] and added to high-quality images.  
  
The train and validation dataset are structured as below:
- Berkeley Segmentation Dataset (https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- Waterloo Exploration Database (https://kedema.org/project/exploration/index.html)

The test dataset is structured as below:
- CBSD68
- Kodak24
- Set5
- Urban100

## Code description
### Models
The GTNN configuration model consists of `UNet_backbone.py` and `Enhancement_network.py`.
- `UNet_backbone.py` includes Noise Estimation Network and Generator Network.
- `Enhancment_network.py` includes U-Net and Swin transformer based Enhancement network.
### Util for training
- `utils.py` contains basic contents required for training and evaluation, such as PSNR and SSIM calculation, and weight initialization.
- `dataset.py` connects training images to Dataloader and changes them so that the model can train.
### Training and testing
- `train_GTNN.py` and `test_GTNN.py` are responsible for training GTNN and evaluating the trained model, respectively.

## Saved model
The `/pretrained model/` folder contains the trained models that constitute the GTNN. The trained models are saved as `.pth` file.

## Evaluation
  
![fig 5](https://github.com/user-attachments/assets/b83ac977-e3d6-4f2f-89fc-dd245c42c02b)  
**Fig. 1.** Visual comparison for a noisy image from Kodak24 dataset with various noise levels. The top row presents the ground truth image and a zoomed-in version of the section enclosed in the red box. Column (a) represents the input with various noise (𝜎 = 50, 75, 90, 110) having PSNR values of (14.59dB/11.73dB/10.88dB/9.65dB), (b) presents the denoising results of DnCNN (PSNR: 26.95dB/24.57dB/22.83dB/21.83dB), (c) shows the denoising performance of FFDNet (PSNR: 27.41dB/24.51dB/22.69dB/21.28dB), (d) visual analysis of DudeNet (PSNR: 27.62dB/25.12dB/23.19dB/22.30dB), (e) NIFBGDNet (PSNR: 27.34𝑑B/24.87dB/23.07dB/22.05dB), and (f) the proposed GTNN (PSNR: 28.20dB/26.45dB/26.98dB/24.68dB).  

![fig 7](https://github.com/user-attachments/assets/2649bdbb-9954-4eb6-af1d-4a94798ca526)  
**Fig. 2.** Visual comparison for denoising "Bird" image from Set5 dataset for noise levels (𝜎 = 50, 75, 90, 110). Input PSNR (dB): (15.48/12.47/11.27/10.04). Row (a) shows the noisy input image under various noise levels, (b) denoising using DnCNN, (c) FFDNet, (d) DudeNet, (e) NIFGDNet, and (f) the proposed GTNN.
