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
- Berkeley Segmentation Dataset 
- Waterloo Exploration Database 

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
![fig 4](https://github.com/user-attachments/assets/80fd5865-0ae4-4f0f-9a7c-d80fc7371900)
![table 3](https://github.com/user-attachments/assets/78cd59ac-a26e-4161-b94b-f7dda3e06901)
![table 4](https://github.com/user-attachments/assets/c6202c31-cbdd-4064-aa2d-98102f892b6c)
![fig 6](https://github.com/user-attachments/assets/77e2ea93-b112-4720-bcf7-0ef797187f75)
![table 5](https://github.com/user-attachments/assets/b89ceb64-ea79-49c3-b002-9fe44b139982)
![table 6](https://github.com/user-attachments/assets/d7c84772-abd1-41ab-9a04-9b0a67482006)
![fig 5](https://github.com/user-attachments/assets/119fdff7-5835-4262-9ca4-f7a644572bf7)
