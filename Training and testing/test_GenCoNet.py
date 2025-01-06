import os
from torch.autograd import Variable
from UNet_backbone import UNet_only4, UNet_only5
from Enhancment_network import SCUNet
from utils import *
from PIL import Image
import torchvision.transforms as transforms

std = 110
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device_ids = [0]

GAN = UNet_only4(in_channels=3)
GAN_model3 = nn.DataParallel(GAN, device_ids=device_ids).cuda()
GAN_model3.load_state_dict(torch.load(os.path.join('pretrain/Gen_3rd.pth')))

GAN1 = UNet_only4(in_channels=3)
GAN_model1 = nn.DataParallel(GAN1, device_ids=device_ids).cuda()
GAN_model1.load_state_dict(torch.load(os.path.join('pretrain/Gen_1st.pth')))

unet = UNet_only4(in_channels=3)
unet_model1 = nn.DataParallel(unet, device_ids=device_ids).cuda()
unet_model1.load_state_dict(torch.load(os.path.join('pretrain/U_net_1st.pth')))

unet2 = UNet_only5(in_channels=6)
unet_model2 = nn.DataParallel(unet2, device_ids=device_ids).cuda()
unet_model2.load_state_dict(torch.load(os.path.join('pretrain/U_net_2nd.pth')))

enh = SCUNet(in_nc=12, config=[4,4,4,4,4,4,4], dim=64)
enh = nn.DataParallel(enh, device_ids=device_ids).cuda()
enh.load_state_dict(torch.load(os.path.join('pretrain/Enh.pth')))

GAN_model3.eval()
unet_model1.eval()
unet_model2.eval()
GAN_model1.eval()
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
    save_dir_R = os.path.join(save_R, 'restored image.png')
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
        esti_noise = unet_model1(test_img)
        u_img1 = torch.clamp(test_img - esti_noise, 0., 1)
        u_img2 = torch.clamp(test_img - unet_model2(u_img1, esti_noise), 0., 1)
        g_img1 = GAN_model1(test_img)
        g_img3 = GAN_model3(u_img2)
        restored_img = torch.clamp(enh(test_img, g_img1, u_img2, g_img3),0,1)

    r_psnr = batch_PSNR(restored_img, GT, 1.)
    r_ssim = batch_SSIM(restored_img, GT)
    print("%s E_PSNR %f   E_SSIM %f" % (r_psnr, r_ssim))

    res_view = np.squeeze(restored_img)
    res_view = res_view.cpu().data.numpy().transpose(1, 2, 0)
    res_view = cv2.cvtColor(res_view, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_dir_R, res_view * 255)

if __name__ == "__main__":
    main()