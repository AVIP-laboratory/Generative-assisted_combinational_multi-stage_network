import os
from torch.autograd import Variable
from UNet_backbone import UNet_only4, UNet_only5
from DE_block import SCUNet
from utils import *
from PIL import Image
import torchvision.transforms as transforms

std = 110
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device_ids = [0]

Gen2 = UNet_only4(in_channels=3)
IT_block2 = nn.DataParallel(Gen2, device_ids=device_ids).cuda()
IT_block2.load_state_dict(torch.load(os.path.join('Saved model/Gen_2nd.pth')))

Gen1 = UNet_only4(in_channels=3)
IT_block1 = nn.DataParallel(Gen1, device_ids=device_ids).cuda()
IT_block1.load_state_dict(torch.load(os.path.join('Saved model/Gen_1st.pth')))

unet = UNet_only4(in_channels=3)
NE_block1 = nn.DataParallel(unet, device_ids=device_ids).cuda()
NE_block1.load_state_dict(torch.load(os.path.join('Saved model/U_net_1st.pth')))

unet2 = UNet_only5(in_channels=6)
NE_block2 = nn.DataParallel(unet2, device_ids=device_ids).cuda()
NE_block2.load_state_dict(torch.load(os.path.join('Saved model/U_net_2nd.pth')))

enh = SCUNet(in_nc=12, config=[4,4,4,4,4,4,4], dim=64)
enh = nn.DataParallel(enh, device_ids=device_ids).cuda()
enh.load_state_dict(torch.load(os.path.join('Saved model/Enh.pth')))

IT_block2.eval()
NE_block1.eval()
NE_block2.eval()
IT_block1.eval()
enh.eval()

def transformss(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    data = transform(image)
    return data

def main():
    save_R = r'save restored image'
    os.makedirs(save_R, exist_ok=True)
    save_dir_DE = os.path.join(save_R, 'restored image.png')
    gt_dir = r'test dataset'

    GT = Image.open(gt_dir).convert('RGB')
    GT = transformss(GT)
    GT = GT.unsqueeze(0)

    noise = torch.zeros(GT.size())
    for nx in range(noise.size()[0]):
        sizen = noise[0, :, :, :].size()
        noise[nx, :, :, :] = torch.FloatTensor(sizen).normal_(mean=0, std=std / 255)

    imgn_test = noise + GT
    test_img = Variable(imgn_test.cuda())
    GT = Variable(GT.cuda())

    with torch.no_grad():
        esti_noise = NE_block1(test_img)
        NE_img1 = torch.clamp(test_img - esti_noise, 0., 1)
        NE_img2 = torch.clamp(test_img - NE_block2(NE_img1, esti_noise), 0., 1)
        IT_img1 = IT_block1(test_img)
        IT_img3 = IT_block2(NE_img2)
        restored_img = torch.clamp(enh(test_img, IT_img1, NE_img2, IT_img3),0,1)

    DE_psnr = batch_PSNR(restored_img, GT, 1.)
    DE_ssim = batch_SSIM(restored_img, GT)
    print("%s DE_PSNR %f   DE_SSIM %f" % (DE_psnr, DE_ssim))

    DE_image = np.squeeze(restored_img)
    DE_image = DE_image.cpu().data.numpy().transpose(1, 2, 0)
    DE_image = cv2.cvtColor(DE_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_dir_DE, DE_image * 255)

if __name__ == "__main__":
    main()