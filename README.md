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
![table 2](https://github.com/user-attachments/assets/1cd43a65-b066-40dd-a5cd-190b1ce7fb95)
![fig 4](https://github.com/user-attachments/assets/09ef1322-6658-4425-83c8-44db111bef63)
![table 3](https://github.com/user-attachments/assets/70c6f556-4871-4b50-9588-35f81df20154)
![table 4](https://github.com/user-attachments/assets/d8425cc7-1c78-4662-8ba2-cc2cdff6a523)
![fig 5](https://github.com/user-attachments/assets/b83ac977-e3d6-4f2f-89fc-dd245c42c02b)
![fig 6](https://github.com/user-attachments/assets/02525782-9646-4f4c-be16-2708613ad38d)
![table 5](https://github.com/user-attachments/assets/41cf301c-0dc5-448a-9d01-f124eb532790)
![table 6](https://github.com/user-attachments/assets/602e93d4-652f-465c-8f91-2c1b0be57d50)
![fig 7](https://github.com/user-attachments/assets/2649bdbb-9954-4eb6-af1d-4a94798ca526)
