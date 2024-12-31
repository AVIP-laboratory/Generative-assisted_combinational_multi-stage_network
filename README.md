# Generative tandem neural network
Noise degrades image quality and can result in the loss of important information, making its removal or minimization essential. However, as noise levels increase, eliminating it becomes exponentially more challenging. This project proposes a **Generative tandem neural network (GTNN)** capable of restoring the textural details and patterns of original images lost due to extreme noise. We will keep you informed with additional details about this project.
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
- `UNet_backbone.py` includes Noise Estimatior block and Generative transloator block.
- `Enhancment_network.py` includes U-Net and Swin transformer based Channel-fusion enhancer block.
### Util for training
- `utils.py` contains basic contents required for training and evaluation, such as PSNR and SSIM calculation, and weight initialization.
- `dataset.py` connects training images to Dataloader and changes them so that the model can train.
### Training and testing
- `train_GTNN.py` and `test_GTNN.py` are responsible for training GTNN and evaluating the trained model, respectively.

## Saved model
The `/trained model/` folder contains the trained models that constitute the GTNN. The trained models are saved as `.pth` file.

## Evaluation
  
![fig 5](https://github.com/user-attachments/assets/b83ac977-e3d6-4f2f-89fc-dd245c42c02b)  
**Fig. 1.** Visual comparison for a noisy image from Kodak24 dataset with various noise levels. (a) The input with varying noise: 𝜎 = 50, 75, 90, 110 (PSNR = 14.59, 11.73, 10.88, 9.65 dB). (b) The denoising results of DnCNN (PSNR = 26.95, 24.57, 22.83, 21.83 dB), (c) FFDNet (PSNR = 27.41, 24.51, 22.69, 21.28 dB), (d) DudeNet (PSNR = 27.62, 25.12, 23.19, 22.30 dB), (e) NIFBGDNet (PSNR = 27.34, 24.87, 23.07, 22.05 dB), and (f) the proposed GTNN (PSNR = 28.20, 26.45, 26.98, 24.68 dB).

![fig 7](https://github.com/user-attachments/assets/2649bdbb-9954-4eb6-af1d-4a94798ca526)  
**Fig. 2.** Visual comparison for noisy ‘Bird’ image from Set5 dataset with various noise levels. (a) The input with varying noise: 𝜎 = 50, 75, 90, 110 (PSNR = 15.48, 12.47, 11.27, 10.04 dB). (b) The denoising results of DnCNN, (PSNR = 25.05, 21.55, 19.84, 18.13 dB), (c) FFDNet (PSNR = 25.43, 21.62, 19.90, 17.98 dB), (d) DudeNet (PSNR =25.04, 21.62, 20.03, 18.30 dB), (e) NIFGDNet (PSNR = 25.42, 21.82, 20.29, 18.50 dB), and (f) the proposed GTNN (29.05, 27.16, 26.27, 25.18 dB).
