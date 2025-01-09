import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import optim
from UNet_backbone import Discriminator, UNet_only4, UNet_only5
from Enhancment_network import SCUNet
from dataset import add_nosie_data
from torch.autograd import Variable
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# transforms 정의
noiseIntL = [10, 110]
val_noiseL = [110, 110]
def transformss(image):
    transform = transforms.Compose([
                        transforms.ToTensor(),
    ])
    data = transform(image)
    return data

print("Loading dataset \n")
train_ds = add_nosie_data(r'\train', transform=transformss)
val_ds = add_nosie_data(r'\valid', transform=transformss)
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)

device_ids = [0]
unet = UNet_only4(in_channels=3)
unet.apply(weights_init_kaiming)
unet_model1 = nn.DataParallel(unet, device_ids=device_ids).cuda()

unet2 = UNet_only5(in_channels=6)
unet2.apply(weights_init_kaiming)
unet_model2 = nn.DataParallel(unet2, device_ids=device_ids).cuda()

GAN = UNet_only4(in_channels=3)
GAN.apply(weights_init_kaiming)
GAN_model3 = nn.DataParallel(GAN, device_ids=device_ids).cuda()

dis = Discriminator(in_channels=3)
dis.apply(weights_init_kaiming)
dis_model3 = nn.DataParallel(dis, device_ids=device_ids).cuda()

GAN1 = UNet_only4(in_channels=3)
GAN1.apply(weights_init_kaiming)
GAN_model1 = nn.DataParallel(GAN1, device_ids=device_ids).cuda()

dis1 = Discriminator(in_channels=3)
dis1.apply(weights_init_kaiming)
dis_model1 = nn.DataParallel(dis1, device_ids=device_ids).cuda()

enh = SCUNet(in_nc=12, config=[4,4,4,4,4,4,4], dim=64)
enh_model = nn.DataParallel(enh, device_ids=device_ids).cuda()

# loss function
criterion = nn.MSELoss().cuda()
loss_func_gan = nn.BCELoss().cuda()
loss_func_pix = nn.L1Loss().cuda()
enh_mse = nn.L1Loss().cuda()
lambda_pixel = 100

# Optimizer
lr = 1e-3
glr = 2e-4
glr2 = 2e-9
lr2 = 1e-9
unet_optimizer1 = optim.Adam(unet_model1.parameters(), lr=lr)
unet_optimizer2 = optim.Adam(unet_model2.parameters(), lr=lr2)
gan_optimizer3 = optim.Adam(GAN_model3.parameters(), lr=glr2)
dis_optimizer3 = optim.Adam(dis_model3.parameters(), lr=glr2)
gan_optimizer1 = optim.Adam(GAN_model1.parameters(), lr=glr)
dis_optimizer1 = optim.Adam(dis_model1.parameters(), lr=glr)
enh_optimizer = optim.Adam(enh_model.parameters(), lr=lr2)

# patch size for cGAN
patch = (1,55, 55)

# Temporary save loss
unet_loss_his1 = []
unet_loss_his2 = []
gen_loss_his1 = []
dis_loss_his1 = []
gen_loss_his3 = []
dis_loss_his3 = []
enh_loss_his = []

val_unet_loss_his1 = []
val_unet_loss_his2 = []

val_gen_loss_his1 = []
val_dis_loss_his1 = []
val_gen_loss_his3 = []
val_dis_loss_his3 = []
val_enh_loss_his = []

unet_psnr_his1 = []
unet_psnr_his2 = []
gen_psnr_his1 = []
gen_psnr_his3 = []
enh_psnr_his = []

val_unet_psnr_his1 = []
val_unet_psnr_his2 = []
val_gen_psnr_his1 = []
val_gen_psnr_his3 = []
val_enh_psnr_his = []

milstone = [100, 150, 250, 300, 400, 450, 550, 600]
num_epochs = 600

print("Start Train")
for epoch in range(num_epochs):
    if epoch < milstone[0]:
        current_lr = lr
        current_glr = glr
    elif epoch >= milstone[0] and epoch < milstone[1]:
        current_lr = lr / 10
        current_glr = glr
    else:
        current_lr = lr / 10000000.
        current_glr = glr
    if epoch >= milstone[1] and epoch < milstone[2]:
        current_lr2 = lr2 * 1000000
    elif epoch >= milstone[2] and epoch < milstone[3]:
        current_lr2 = lr2 * 100000
    else:
        current_lr2 = lr2
    if epoch >= milstone[3] and epoch < milstone[4]:
        current_glr3 = glr2 * 100000
    elif epoch >= milstone[4] and epoch < milstone[5]:
        current_glr3 = glr2 * 100000
    else:
        current_glr3 = glr2
    if epoch >= milstone[5] and epoch < milstone[6]:
        enh_lr = lr2 * 1e+5
    elif epoch >= milstone[6] and epoch < milstone[7]:
        enh_lr = lr2 * 1e+5
    else:
        enh_lr = lr2

    # set learning rate
    for param_group in unet_optimizer1.param_groups:
        param_group["lr"] = current_lr

    for param_group in unet_optimizer2.param_groups:
        param_group["lr"] = current_lr2

    for param_group in gan_optimizer3.param_groups:
        param_group["lr"] = current_glr3
    for param_group in dis_optimizer3.param_groups:
        param_group["lr"] = current_glr3

    for param_group in gan_optimizer1.param_groups:
        param_group["lr"] = current_glr
    for param_group in dis_optimizer1.param_groups:
        param_group["lr"] = current_glr

    for param_group in enh_optimizer.param_groups:
        param_group["lr"] = enh_lr

    print('U1 learning rate %f   U2 learning rate %f' % (current_lr, current_lr2))
    print('G1 learning rate %f   G2 learning rate %f' % (current_glr, current_glr3))
    print('E1 learning rate %f' % (enh_lr))

    val_gen_psnr_temp = 0
    val_unet_psnr_temp = 0
    val_enh_psnr_temp = 0

    val_gen_psnr_temp1 = 0

    val_unet_loss_temp1 = 0
    val_unet_loss_temp2 = 0
    val_dis_loss_temp1 = 0
    val_dis_loss_temp3 = 0
    val_gen_loss_temp1 = 0
    val_gen_loss_temp3 = 0
    val_enh_loss_temp = 0

    unet_loss_temp1 = 0
    unet_loss_temp2 = 0

    gen_loss_temp3 = 0
    dis_loss_temp3 = 0
    gen_loss_temp1 = 0
    dis_loss_temp1 = 0
    enh_loss_temp = 0

    unet_psnr_temp1 = 0
    unet_psnr_temp2 = 0
    gen_psnr_temp1 = 0
    gen_psnr_temp3 = 0
    enh_psnr_temp = 0

    batch_count = 0
    i = 0
    for train_image in train_dl:
        # training step
        unet_model1.train()
        unet_model1.zero_grad()
        unet_optimizer1.zero_grad()

        unet_model2.train()
        unet_model2.zero_grad()
        unet_optimizer2.zero_grad()

        GAN_model3.train()
        GAN_model3.zero_grad()
        gan_optimizer3.zero_grad()

        dis_model3.train()
        dis_model3.zero_grad()
        dis_optimizer3.zero_grad()

        GAN_model1.train()
        GAN_model1.zero_grad()
        gan_optimizer1.zero_grad()

        dis_model1.train()
        dis_model1.zero_grad()
        dis_optimizer1.zero_grad()

        enh_model.train()
        enh_model.zero_grad()
        enh_optimizer.zero_grad()

        #==============================Prepare train data======================================

        img_train = train_image

        ba_si = img_train.size(0)
        real_label = torch.ones(ba_si, *patch, requires_grad=False).cuda()
        fake_label = torch.zeros(ba_si, *patch, requires_grad=False).cuda()

        noise = torch.zeros(img_train.size())
        stdN = np.random.uniform(noiseIntL[0], noiseIntL[1], size=noise.size()[0])
        for n in range(noise.size()[0]):
            sizeN = noise[0, :, :, :].size()
            noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)
        imgn_train = img_train + noise

        # Create input Variables
        img_train = Variable(img_train.cuda())
        imgn_train = Variable(imgn_train.cuda())
        noise = Variable(noise.cuda())

        # ============== Denoising 1 ==============
        esti_noise1 = unet_model1(imgn_train)
        unet_loss1 = criterion(esti_noise1, noise) / (imgn_train.size()[0] * 2)
        unet_loss1.backward()
        unet_optimizer1.step()

        out_unet1 = torch.clamp(imgn_train - unet_model1(imgn_train), 0., 1.)

        # ============== Generator1 train ==============

        fake_b1 = GAN_model1(imgn_train)
        out_dis1 = dis_model1(fake_b1, img_train)

        gen_loss1 = loss_func_gan(out_dis1, real_label)
        pixel_loss1 = loss_func_pix(fake_b1, img_train)

        g_loss1 = gen_loss1 + lambda_pixel * pixel_loss1
        g_loss1.backward()
        gan_optimizer1.step()

        # ============== discriminator1 ==============
        out_dis1 = dis_model1(img_train, imgn_train)
        real_loss1 = loss_func_gan(out_dis1, real_label)

        out_dis1 = dis_model1(fake_b1.detach(), imgn_train)
        fake_loss1 = loss_func_gan(out_dis1, fake_label)

        d_loss1 = (real_loss1 + fake_loss1) / 2.
        d_loss1.backward()
        dis_optimizer1.step()

        # ============== Denoising 2 ==============
        imgn_train2 = torch.clamp(out_unet1.detach() - img_train.detach(), 0, 1)
        esti_noise2 = unet_model2(out_unet1.detach(), esti_noise1.detach())
        unet_loss2 = criterion(esti_noise2, noise.detach()) / (imgn_train.size()[0] * 2)
        unet_loss2.backward()
        unet_optimizer2.step()

        out_unet2 = torch.clamp(imgn_train.detach() - esti_noise2, 0., 1.)

        # ============== Generator3 train ==============
        unet_model1.eval()
        unet_model2.eval()
        out_unet2 = out_unet2.detach()

        fake_b3 = GAN_model3(out_unet2)
        out_dis3 = dis_model3(fake_b3, img_train)

        gen_loss3 = loss_func_gan(out_dis3, real_label)
        pixel_loss3 = loss_func_pix(fake_b3, img_train)

        g_loss3 = gen_loss3 + lambda_pixel * pixel_loss3
        g_loss3.backward()
        gan_optimizer3.step()

        # ============== discriminator3 ==============
        out_dis3 = dis_model3(img_train, out_unet2)
        real_loss3 = loss_func_gan(out_dis3, real_label)

        out_dis3 = dis_model3(fake_b3.detach(), out_unet2)
        fake_loss3 = loss_func_gan(out_dis3, fake_label)

        d_loss3 = (real_loss3 + fake_loss3) / 2.
        d_loss3.backward()
        dis_optimizer3.step()

        # ============== Enhancing train ==============
        restored_image = enh_model(imgn_train, fake_b1.detach(), out_unet2.detach(), fake_b3.detach())
        enh_loss = enh_mse(restored_image, img_train) / (imgn_train.size()[0] * 2)
        enh_loss.backward()
        enh_optimizer.step()

        # ============== results ==============
        U_psnr1 = batch_PSNR(out_unet1, img_train, 1.)
        U_psnr2 = batch_PSNR(out_unet2, img_train, 1.)
        G_psnr1 = batch_PSNR(fake_b1, img_train, 1.)
        G_psnr3 = batch_PSNR(fake_b3, img_train, 1.)
        E_psnr = batch_PSNR(restored_image, img_train, 1.)

        unet_loss_temp1 += unet_loss1.item()
        unet_loss_temp2 += unet_loss2.item()
        gen_loss_temp1 += g_loss1.item()
        dis_loss_temp1 += d_loss1.item()
        gen_loss_temp3 += g_loss3.item()
        dis_loss_temp3 += d_loss3.item()
        enh_loss_temp += enh_loss.item()
        unet_psnr_temp1 += U_psnr1
        unet_psnr_temp2 += U_psnr2
        gen_psnr_temp1 += G_psnr1
        gen_psnr_temp3 += G_psnr3
        enh_psnr_temp += E_psnr

        if (i + 1) % 30 == 0:
            print("[epoch %d][%d/%d] U_loss1: %.4f U_loss2: %.4f \t\t\t U_PSNR1: %.4f U_PSNR2: %.4f" %
                  (epoch + 1, i + 1, len(train_dl), unet_loss1.item(), unet_loss2.item(), U_psnr1, U_psnr2))
            print("\t\t\t\t G_loss1: %.4f G_loss3: %.4f \t\tG_PSNR1: %.4f G_PSNR2: %.4f" %
                  (g_loss1.item(), g_loss3.item(), G_psnr1, G_psnr3))
            print("\t\t\t\t E_loss1: %.4f \t\t\t\t\t\t\t\t\t\t E_PSNR: %.4f" %
                  (enh_loss.item(), E_psnr))

        i += 1
        batch_count += 1

    # Save train loss, psnr
    unet_loss_his1.append(unet_loss_temp1 / 540)
    unet_loss_his2.append(unet_loss_temp2 / 540)
    gen_loss_his3.append(gen_loss_temp3 / 540)
    dis_loss_his3.append(dis_loss_temp3 / 540)
    gen_loss_his1.append(gen_loss_temp1 / 540)
    dis_loss_his1.append(dis_loss_temp1 / 540)
    enh_loss_his.append(enh_loss_temp / 540)

    unet_psnr_his1.append(unet_psnr_temp1 / 540)
    unet_psnr_his2.append(unet_psnr_temp2 / 540)
    gen_psnr_his1.append(gen_psnr_temp1 / 540)
    gen_psnr_his3.append(gen_psnr_temp3 / 540)
    enh_psnr_his.append(enh_psnr_temp / 540)

    # Validation
    unet_model1.eval()
    unet_model2.eval()
    dis_model1.eval()
    GAN_model1.eval()
    dis_model3.eval()
    GAN_model3.eval()
    enh_model.eval()

    psnr_val_u1 = 0
    psnr_val_u2 = 0
    psnr_val_g1 = 0
    psnr_val_g3 = 0
    psnr_val_e = 0
    val_count = 0
    for k in range(len(val_ds)):
        img_val = torch.unsqueeze(val_ds[k], 0)
        stdN = np.random.uniform(noiseIntL[0], noiseIntL[1])
        noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=stdN / 255.)
        imgn_val = img_val + noise
        img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
        noise = Variable(noise.cuda(), volatile=True)

        est_noise_val1 = unet_model1(imgn_val)
        unet_val1 = torch.clamp(imgn_val - est_noise_val1, 0., 1.)
        gen_val1 = GAN_model1(imgn_val)
        est_noise_val2 = unet_model2(unet_val1.detach(), est_noise_val1.detach())
        unet_val2 = torch.clamp(imgn_val - est_noise_val2, 0, 1)
        gen_val3 = GAN_model3(unet_val2)
        enh_val = enh_model(imgn_val, gen_val1, unet_val2, gen_val3)

        # Restoring validation PSNR
        psnr_val_u1 += batch_PSNR(unet_val1, img_val, 1.)
        psnr_val_u2 += batch_PSNR(unet_val2, img_val, 1.)
        psnr_val_g1 += batch_PSNR(gen_val1, img_val, 1.)
        psnr_val_g3 += batch_PSNR(gen_val3, img_val, 1.)
        psnr_val_e += batch_PSNR(enh_val, img_val, 1.)

        # Restoring validation loss
        val_unet_loss1 = criterion(est_noise_val1, noise) / (imgn_val.size()[0] * 2)
        val_unet_loss_temp1 += val_unet_loss1.item()
        val_unet_loss2 = criterion(est_noise_val2, noise) / (imgn_val.size()[0] * 2)
        val_unet_loss_temp2 += val_unet_loss2.item()

        ba_si = img_val.size(0)
        real_label = torch.ones(ba_si, *patch, requires_grad=False).cuda()
        fake_label = torch.zeros(ba_si, *patch, requires_grad=False).cuda()

        val_out_dis1 = dis_model3(gen_val1, img_val)
        val_gen_loss1 = loss_func_gan(val_out_dis1, real_label)
        val_pix_loss1 = loss_func_pix(gen_val1, img_val)

        val_gen_loss1 = (val_gen_loss1 + lambda_pixel * val_pix_loss1) / (imgn_val.size()[0] * 2)
        val_dis_loss_temp1 += val_gen_loss1.item()

        val_out_dis3 = dis_model3(gen_val3, img_val)
        val_gen_loss3 = loss_func_gan(val_out_dis3, real_label)
        val_pix_loss3 = loss_func_pix(gen_val3, img_val)

        val_gen_loss3 = (val_gen_loss3 + lambda_pixel * val_pix_loss3) / (imgn_val.size()[0] * 2)
        val_dis_loss_temp3 += val_gen_loss3.item()

        val_enh_loss = enh_mse(enh_val, img_val) / (imgn_val.size()[0] * 2)
        val_enh_loss_temp += val_enh_loss.item()

        val_count = val_count + 1

    val_unet_loss_his1.append(val_unet_loss_temp1 / len(val_ds))
    val_unet_loss_his2.append(val_unet_loss_temp2 / len(val_ds))
    val_gen_loss_his3.append(val_dis_loss_temp3 / len(val_ds))
    val_gen_loss_his1.append(val_dis_loss_temp1 / len(val_ds))
    val_enh_loss_his.append(val_enh_loss_temp / len(val_ds))

    psnr_val_u1 /= len(val_ds)
    psnr_val_u2 /= len(val_ds)
    psnr_val_g1 /= len(val_ds)
    psnr_val_g3 /= len(val_ds)
    psnr_val_e /= len(val_ds)

    val_unet_psnr_his1.append(psnr_val_u1)
    val_unet_psnr_his2.append(psnr_val_u2)
    val_gen_psnr_his1.append(psnr_val_g1)
    val_gen_psnr_his3.append(psnr_val_g3)
    val_enh_psnr_his.append(psnr_val_e)

    print("\n[epoch %d] PSNR_U1: %.4f PSNR_U2: %.4f" % (epoch + 1, psnr_val_u1, psnr_val_u2))
    print("\t\t PSNR_G1: %.4f PSNR_G3: %.4f" % (psnr_val_g1, psnr_val_g3))
    print("\t\t PSNR_E: %.4f" % (psnr_val_e))

# save model
path2models = './GTNN/'
os.makedirs(path2models, exist_ok=True)

path2weights_U1 = os.path.join(path2models, 'U_net_1st.pth')
path2weights_U2 = os.path.join(path2models, 'U_net_2nd.pth')
path2weights_G1 = os.path.join(path2models, 'Gen_1st.pth')
path2weights_G3 = os.path.join(path2models, 'Gen_2nd.pth')
path2weights_E = os.path.join(path2models, 'Enh.pth')
#
torch.save(unet_model1.state_dict(), path2weights_U1)
torch.save(unet_model2.state_dict(), path2weights_U2)
torch.save(GAN_model1.state_dict(), path2weights_G1)
torch.save(GAN_model3.state_dict(), path2weights_G3)
torch.save(enh_model.state_dict(), path2weights_E)


# plot loss =====================
plt.figure(figsize=(7, 5))
plt.title('Train Loss Progress')
plt.plot(unet_loss_his1, label='U. Loss1')
plt.plot(unet_loss_his2, label='U. Loss2')
plt.plot(enh_loss_his, label='GTNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.title('Train Loss Progress')
plt.plot(gen_loss_his1, label='G. Loss1')
plt.plot(dis_loss_his1, label='D. Loss1')
plt.plot(gen_loss_his3, label='G. Loss3')
plt.plot(dis_loss_his3, label='D. Loss3')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.title('Val Loss Progress')
plt.plot(val_unet_loss_his1, label='U. Loss1')
plt.plot(val_unet_loss_his2, label='U. Loss2')
plt.plot(val_enh_loss_his, label='GTNN. Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.title('Val Loss Progress')
plt.plot(val_gen_loss_his1, label='G. Loss1')
plt.plot(val_gen_loss_his3, label='G. Loss3')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.title('Train PSNR Progress')
plt.plot(unet_psnr_his1, label='U. PSNR1')
plt.plot(unet_psnr_his2, label='U. PSNR2')
plt.plot(gen_psnr_his1, label='G. PSNR1')
plt.plot(gen_psnr_his3, label='G. PSNR3')
plt.plot(enh_psnr_his, label='GTNN PSNR')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.title('Val PSNR Progress')
plt.plot(val_unet_psnr_his1, label='U. PSNR1')
plt.plot(val_unet_psnr_his2, label='U. PSNR2')
plt.plot(val_gen_psnr_his1, label='G. PSNR1')
plt.plot(val_gen_psnr_his3, label='G. PSNR2')
plt.plot(val_enh_psnr_his, label='GTNN PSNR')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save train, valid log
path2loss = './GTNN loss/'
os.makedirs(path2loss, exist_ok=True)
np.savetxt(os.path.join(path2loss, 'val_unet_loss_1st.txt'), val_unet_loss_his1)
np.savetxt(os.path.join(path2loss, 'val_unet_loss_2nd.txt'), val_unet_loss_his2)
np.savetxt(os.path.join(path2loss, 'val_gen_loss_1st.txt'), val_gen_loss_his1)
np.savetxt(os.path.join(path2loss, 'val_gen_loss_3rd.txt'), val_gen_loss_his3)
np.savetxt(os.path.join(path2loss, 'val_res_loss.txt'), val_enh_loss_his)

np.savetxt(os.path.join(path2loss, 'unet_loss_1st.txt'), unet_loss_his1)
np.savetxt(os.path.join(path2loss, 'unet_loss_2nd.txt'), unet_loss_his2)
np.savetxt(os.path.join(path2loss, 'gen_loss_1st.txt'), gen_loss_his1)
np.savetxt(os.path.join(path2loss, 'dis_loss_1st.txt'), dis_loss_his1)
np.savetxt(os.path.join(path2loss, 'gen_loss_3rd.txt'), gen_loss_his3)
np.savetxt(os.path.join(path2loss, 'dis_loss_3rd.txt'), dis_loss_his3)
np.savetxt(os.path.join(path2loss, 'res_loss.txt'), enh_loss_his)

np.savetxt(os.path.join(path2loss, 'unet_psnr_1st.txt'), unet_psnr_his1)
np.savetxt(os.path.join(path2loss, 'unet_psnr_2nd.txt'), unet_psnr_his2)
np.savetxt(os.path.join(path2loss, 'gen_psnr_1st.txt'), gen_psnr_his1)
np.savetxt(os.path.join(path2loss, 'gen_psnr_3rd.txt'), gen_psnr_his3)
np.savetxt(os.path.join(path2loss, 'res_psnr.txt'), enh_psnr_his)

np.savetxt(os.path.join(path2loss, 'val_unet_psnr_1st.txt'), val_unet_psnr_his1)
np.savetxt(os.path.join(path2loss, 'val_unet_psnr_2nd.txt'), val_unet_psnr_his2)
np.savetxt(os.path.join(path2loss, 'val_gen_psnr_1st.txt'), val_gen_psnr_his1)
np.savetxt(os.path.join(path2loss, 'val_gen_psnr_3rd.txt'), val_gen_psnr_his3)
np.savetxt(os.path.join(path2loss, 'val_res_psnr.txt'), val_enh_psnr_his)