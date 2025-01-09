import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import optim
from UNet_backbone import Discriminator, UNet_only4, UNet_only5
from DE_block import SCUNet
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
NE_block1 = nn.DataParallel(unet, device_ids=device_ids).cuda()

unet2 = UNet_only5(in_channels=6)
unet2.apply(weights_init_kaiming)
NE_block2 = nn.DataParallel(unet2, device_ids=device_ids).cuda()

GAN = UNet_only4(in_channels=3)
GAN.apply(weights_init_kaiming)
GAN_block2 = nn.DataParallel(GAN, device_ids=device_ids).cuda()

dis = Discriminator(in_channels=3)
dis.apply(weights_init_kaiming)
dis_block2 = nn.DataParallel(dis, device_ids=device_ids).cuda()

GAN1 = UNet_only4(in_channels=3)
GAN1.apply(weights_init_kaiming)
GAN_block1 = nn.DataParallel(GAN1, device_ids=device_ids).cuda()

dis1 = Discriminator(in_channels=3)
dis1.apply(weights_init_kaiming)
dis_block1 = nn.DataParallel(dis1, device_ids=device_ids).cuda()

DE = SCUNet(in_nc=12, config=[4,4,4,4,4,4,4], dim=64)
DE_block = nn.DataParallel(DE, device_ids=device_ids).cuda()

# loss function
criterion = nn.MSELoss().cuda()
loss_func_gan = nn.BCELoss().cuda()
loss_func_pix = nn.L1Loss().cuda()
DE_mse = nn.L1Loss().cuda()
lambda_pixel = 100

# Optimizer
lr = 1e-3
glr = 2e-4
glr2 = 2e-9
lr2 = 1e-9
unet_optimizer1 = optim.Adam(NE_block1.parameters(), lr=lr)
unet_optimizer2 = optim.Adam(NE_block2.parameters(), lr=lr2)
gan_optimizer2 = optim.Adam(GAN_block2.parameters(), lr=glr2)
dis_optimizer2 = optim.Adam(dis_block2.parameters(), lr=glr2)
gan_optimizer1 = optim.Adam(GAN_block1.parameters(), lr=glr)
dis_optimizer1 = optim.Adam(dis_block1.parameters(), lr=glr)
DE_optimizer = optim.Adam(DE_block.parameters(), lr=lr2)

# patch size for cGAN
patch = (1,55, 55)

# Temporary save loss
NE_loss_his1 = []
NE_loss_his2 = []
gen_loss_his1 = []
dis_loss_his1 = []
gen_loss_his3 = []
dis_loss_his3 = []
DE_loss_his = []

val_NE_loss_his1 = []
val_NE_loss_his2 = []

val_gen_loss_his1 = []
val_dis_loss_his1 = []
val_gen_loss_his3 = []
val_dis_loss_his3 = []
val_DE_loss_his = []

NE_psnr_his1 = []
NE_psnr_his2 = []
IT_psnr_his1 = []
IT_psnr_his3 = []
DE_psnr_his = []

val_NE_psnr_his1 = []
val_NE_psnr_his2 = []
val_IT_psnr_his1 = []
val_IT_psnr_his3 = []
val_DE_psnr_his = []

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
        current_lr = lr / 1e-6
        current_glr = glr
    if epoch >= milstone[1] and epoch < milstone[2]:
        current_lr2 = lr2 * 1e+6
    elif epoch >= milstone[2] and epoch < milstone[3]:
        current_lr2 = lr2 * 1e+5
    else:
        current_lr2 = lr2
    if epoch >= milstone[3] and epoch < milstone[4]:
        current_glr2 = glr2 * 1e+5
    elif epoch >= milstone[4] and epoch < milstone[5]:
        current_glr2 = glr2 * 1e+5
    else:
        current_glr2 = glr2
    if epoch >= milstone[5] and epoch < milstone[6]:
        DE_lr = lr2 * 1e+5
    elif epoch >= milstone[6] and epoch < milstone[7]:
        DE_lr = lr2 * 1e+5
    else:
        DE_lr = lr2

    # set learning rate
    for param_group in unet_optimizer1.param_groups:
        param_group["lr"] = current_lr

    for param_group in unet_optimizer2.param_groups:
        param_group["lr"] = current_lr2

    for param_group in gan_optimizer2.param_groups:
        param_group["lr"] = current_glr2
    for param_group in dis_optimizer2.param_groups:
        param_group["lr"] = current_glr2

    for param_group in gan_optimizer1.param_groups:
        param_group["lr"] = current_glr
    for param_group in dis_optimizer1.param_groups:
        param_group["lr"] = current_glr

    for param_group in DE_optimizer.param_groups:
        param_group["lr"] = DE_lr

    print('U1 learning rate %f   U2 learning rate %f' % (current_lr, current_lr2))
    print('G1 learning rate %f   G2 learning rate %f' % (current_glr, current_glr2))
    print('E1 learning rate %f' % (DE_lr))

    val_IT_psnr_temp = 0
    val_NE_psnr_temp = 0
    val_DE_psnr_temp = 0

    val_IT_psnr_temp1 = 0

    val_NE_loss1_temp = 0
    val_NE_loss_temp2 = 0
    val_dis_loss_temp1 = 0
    val_dis_loss_temp2 = 0
    val_gen_loss_temp1 = 0
    val_gen_loss_temp2 = 0
    val_DE_loss_temp = 0

    NE_loss1_temp = 0
    NE_loss_temp2 = 0

    gen_loss_temp2 = 0
    dis_loss_temp2 = 0
    gen_loss_temp1 = 0
    dis_loss_temp1 = 0
    DE_loss_temp = 0

    NE_psnr_temp1 = 0
    NE_psnr_temp2 = 0
    IT_psnr_temp1 = 0
    IT_psnr_temp2 = 0
    DE_psnr_temp = 0

    batch_count = 0
    i = 0
    for train_image in train_dl:
        # training step
        NE_block1.train()
        NE_block1.zero_grad()
        unet_optimizer1.zero_grad()

        NE_block2.train()
        NE_block2.zero_grad()
        unet_optimizer2.zero_grad()

        GAN_block2.train()
        GAN_block2.zero_grad()
        gan_optimizer2.zero_grad()

        dis_block2.train()
        dis_block2.zero_grad()
        dis_optimizer2.zero_grad()

        GAN_block1.train()
        GAN_block1.zero_grad()
        gan_optimizer1.zero_grad()

        dis_block1.train()
        dis_block1.zero_grad()
        dis_optimizer1.zero_grad()

        DE_block.train()
        DE_block.zero_grad()
        DE_optimizer.zero_grad()

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

        # ============== Train 1st NE block ==============
        esti_noise1 = NE_block1(imgn_train)
        NE_loss1 = criterion(esti_noise1, noise) / (imgn_train.size()[0] * 2)
        NE_loss1.backward()
        unet_optimizer1.step()

        out_NE1 = torch.clamp(imgn_train - NE_block1(imgn_train), 0., 1.)

        # ============== Train 1st IT block - Generator ==============

        fake_IT1 = GAN_block1(imgn_train)
        out_dis1 = dis_block1(fake_IT1, img_train)

        gen_loss1 = loss_func_gan(out_dis1, real_label)
        pixel_loss1 = loss_func_pix(fake_IT1, img_train)

        g_loss1 = gen_loss1 + lambda_pixel * pixel_loss1
        g_loss1.backward()
        gan_optimizer1.step()

        # ============== Train 1st IT block - Discriminator ==============
        out_dis1 = dis_block1(img_train, imgn_train)
        real_loss1 = loss_func_gan(out_dis1, real_label)

        out_dis1 = dis_block1(fake_IT1.detach(), imgn_train)
        fake_loss1 = loss_func_gan(out_dis1, fake_label)

        d_loss1 = (real_loss1 + fake_loss1) / 2.
        d_loss1.backward()
        dis_optimizer1.step()

        # ==============  Train 2nd NE block  ==============
        imgn_train2 = torch.clamp(out_NE1.detach() - img_train.detach(), 0, 1)
        esti_noise2 = NE_block2(out_NE1.detach(), esti_noise1.detach())
        NE_loss = criterion(esti_noise2, noise.detach()) / (imgn_train.size()[0] * 2)
        NE_loss.backward()
        unet_optimizer2.step()

        out_NE2 = torch.clamp(imgn_train.detach() - esti_noise2, 0., 1.)

        # ============== Train 2nd IT block - Generator ==============
        NE_block1.eval()
        NE_block2.eval()
        out_NE2 = out_NE2.detach()

        fake_IT2 = GAN_block2(out_NE2)
        out_dis3 = dis_block2(fake_IT2, img_train)

        gen_loss2 = loss_func_gan(out_dis3, real_label)
        pixel_loss2 = loss_func_pix(fake_IT2, img_train)

        g_loss2 = gen_loss2 + lambda_pixel * pixel_loss2
        g_loss2.backward()
        gan_optimizer2.step()

        # ============== Train 2nd IT block - Discriminator ==============
        out_dis2 = dis_block2(img_train, out_NE2)
        real_loss2 = loss_func_gan(out_dis2, real_label)

        out_dis2 = dis_block2(fake_IT2.detach(), out_NE2)
        fake_loss2 = loss_func_gan(out_dis2, fake_label)

        d_loss2 = (real_loss2 + fake_loss2) / 2.
        d_loss2.backward()
        dis_optimizer2.step()

        # ============== Train DE block ==============
        restored_image = DE_block(imgn_train, fake_IT1.detach(), out_NE2.detach(), fake_IT2.detach())
        DE_loss = DE_mse(restored_image, img_train) / (imgn_train.size()[0] * 2)
        DE_loss.backward()
        DE_optimizer.step()

        # ============== results ==============
        NE_psnr1 = batch_PSNR(out_NE1, img_train, 1.)
        NE_psnr2 = batch_PSNR(out_NE2, img_train, 1.)
        IT_psnr1 = batch_PSNR(fake_IT1, img_train, 1.)
        IT_psnr2 = batch_PSNR(fake_IT2, img_train, 1.)
        DE_psnr = batch_PSNR(restored_image, img_train, 1.)

        NE_loss1_temp += NE_loss1.item()
        NE_loss_temp2 += NE_loss.item()
        gen_loss_temp1 += g_loss1.item()
        dis_loss_temp1 += d_loss1.item()
        gen_loss_temp2 += g_loss2.item()
        dis_loss_temp2 += d_loss2.item()
        DE_loss_temp += DE_loss.item()
        NE_psnr_temp1 += NE_psnr1
        NE_psnr_temp2 += NE_psnr2
        IT_psnr_temp1 += IT_psnr1
        IT_psnr_temp2 += IT_psnr2
        DE_psnr_temp += DE_psnr

        if (i + 1) % 30 == 0:
            print("[epoch %d][%d/%d] U_loss1: %.4f U_loss2: %.4f \t\t\t NE_PSNR1: %.4f NE_PSNR2: %.4f" %
                  (epoch + 1, i + 1, len(train_dl), NE_loss1.item(), NE_loss.item(), NE_psnr1, NE_psnr2))
            print("\t\t\t\t IT_loss1: %.4f IT_loss2: %.4f \t\tIT_psnr1: %.4f IT_PSNR2: %.4f" %
                  (g_loss1.item(), g_loss2.item(), IT_psnr1, IT_psnr2))
            print("\t\t\t\t E_loss1: %.4f \t\t\t\t\t\t\t\t\t\t DE_PSNR: %.4f" %
                  (DE_loss.item(), DE_psnr))

        i += 1
        batch_count += 1

    # Save train loss, psnr
    NE_loss_his1.append(NE_loss1_temp / 540)
    NE_loss_his2.append(NE_loss_temp2 / 540)
    gen_loss_his3.append(gen_loss_temp2 / 540)
    dis_loss_his3.append(dis_loss_temp2 / 540)
    gen_loss_his1.append(gen_loss_temp1 / 540)
    dis_loss_his1.append(dis_loss_temp1 / 540)
    DE_loss_his.append(DE_loss_temp / 540)

    NE_psnr_his1.append(NE_psnr_temp1 / 540)
    NE_psnr_his2.append(NE_psnr_temp2 / 540)
    IT_psnr_his1.append(IT_psnr_temp1 / 540)
    IT_psnr_his3.append(IT_psnr_temp2 / 540)
    DE_psnr_his.append(DE_psnr_temp / 540)

    # Validation
    NE_block1.eval()
    NE_block2.eval()
    dis_block1.eval()
    GAN_block1.eval()
    dis_block2.eval()
    GAN_block2.eval()
    DE_block.eval()

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

        est_noise_val1 = NE_block1(imgn_val)
        NE_val1 = torch.clamp(imgn_val - est_noise_val1, 0., 1.)
        IT_val1 = GAN_block1(imgn_val)
        est_noise_val2 = NE_block2(NE_val1.detach(), est_noise_val1.detach())
        NE_val2 = torch.clamp(imgn_val - est_noise_val2, 0, 1)
        IT_val2 = GAN_block2(NE_val2)
        DE_val = DE_block(imgn_val, IT_val1, NE_val2, IT_val2)

        # Restoring validation PSNR
        psnr_val_u1 += batch_PSNR(NE_val1, img_val, 1.)
        psnr_val_u2 += batch_PSNR(NE_val2, img_val, 1.)
        psnr_val_g1 += batch_PSNR(IT_val1, img_val, 1.)
        psnr_val_g3 += batch_PSNR(IT_val2, img_val, 1.)
        psnr_val_e += batch_PSNR(DE_val, img_val, 1.)

        # Restoring validation loss
        val_NE_loss1 = criterion(est_noise_val1, noise) / (imgn_val.size()[0] * 2)
        val_NE_loss1_temp += val_NE_loss1.item()
        val_NE_loss = criterion(est_noise_val2, noise) / (imgn_val.size()[0] * 2)
        val_NE_loss_temp2 += val_NE_loss.item()

        ba_si = img_val.size(0)
        real_label = torch.ones(ba_si, *patch, requires_grad=False).cuda()
        fake_label = torch.zeros(ba_si, *patch, requires_grad=False).cuda()

        val_out_dis1 = dis_block2(IT_val1, img_val)
        val_gen_loss1 = loss_func_gan(val_out_dis1, real_label)
        val_pix_loss1 = loss_func_pix(IT_val1, img_val)

        val_gen_loss1 = (val_gen_loss1 + lambda_pixel * val_pix_loss1) / (imgn_val.size()[0] * 2)
        val_dis_loss_temp1 += val_gen_loss1.item()

        val_out_dis2 = dis_block2(IT_val2, img_val)
        val_gen_loss2 = loss_func_gan(val_out_dis2, real_label)
        val_pix_loss2 = loss_func_pix(IT_val2, img_val)

        val_gen_loss2 = (val_gen_loss2 + lambda_pixel * val_pix_loss2) / (imgn_val.size()[0] * 2)
        val_dis_loss_temp2 += val_gen_loss2.item()

        val_DE_loss = DE_mse(DE_val, img_val) / (imgn_val.size()[0] * 2)
        val_DE_loss_temp += val_DE_loss.item()

        val_count = val_count + 1

    val_NE_loss_his1.append(val_NE_loss1_temp / len(val_ds))
    val_NE_loss_his2.append(val_NE_loss_temp2 / len(val_ds))
    val_gen_loss_his3.append(val_dis_loss_temp2 / len(val_ds))
    val_gen_loss_his1.append(val_dis_loss_temp1 / len(val_ds))
    val_DE_loss_his.append(val_DE_loss_temp / len(val_ds))

    psnr_val_u1 /= len(val_ds)
    psnr_val_u2 /= len(val_ds)
    psnr_val_g1 /= len(val_ds)
    psnr_val_g3 /= len(val_ds)
    psnr_val_e /= len(val_ds)

    val_NE_psnr_his1.append(psnr_val_u1)
    val_NE_psnr_his2.append(psnr_val_u2)
    val_IT_psnr_his1.append(psnr_val_g1)
    val_IT_psnr_his3.append(psnr_val_g3)
    val_DE_psnr_his.append(psnr_val_e)

    print("\n[epoch %d] PSNR_U1: %.4f PSNR_U2: %.4f" % (epoch + 1, psnr_val_u1, psnr_val_u2))
    print("\t\t PSNR_G1: %.4f PSNR_G3: %.4f" % (psnr_val_g1, psnr_val_g3))
    print("\t\t PSNR_E: %.4f" % (psnr_val_e))

# save block
path2blocks = './blocks/'
os.makedirs(path2blocks, exist_ok=True)

path2weights_NE1 = os.path.join(path2blocks, '1st_NE.pth')
path2weights_NE2 = os.path.join(path2blocks, '2nd_NE.pth')
path2weights_IT1 = os.path.join(path2blocks, '1st_IT.pth')
path2weights_IT3 = os.path.join(path2blocks, '2nd_IT.pth')
path2weights_DE = os.path.join(path2blocks, 'DE.pth')
#
torch.save(NE_block1.state_dict(), path2weights_NE1)
torch.save(NE_block2.state_dict(), path2weights_NE2)
torch.save(GAN_block1.state_dict(), path2weights_IT1)
torch.save(GAN_block2.state_dict(), path2weights_IT3)
torch.save(DE_block.state_dict(), path2weights_DE)


# plot loss =====================
plt.figure(figsize=(7, 5))
plt.title('Train Loss Progress')
plt.plot(NE_loss_his1, label='U. Loss1')
plt.plot(NE_loss_his2, label='U. Loss2')
plt.plot(DE_loss_his, label='GainNet Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.title('Train Loss Progress')
plt.plot(gen_loss_his1, label='G. Loss1')
plt.plot(dis_loss_his1, label='D. Loss1')
plt.plot(gen_loss_his3, label='G. loss2')
plt.plot(dis_loss_his3, label='D. loss2')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.title('Val Loss Progress')
plt.plot(val_NE_loss_his1, label='U. Loss1')
plt.plot(val_NE_loss_his2, label='U. Loss2')
plt.plot(val_DE_loss_his, label='GainNet. Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.title('Val Loss Progress')
plt.plot(val_gen_loss_his1, label='G. Loss1')
plt.plot(val_gen_loss_his3, label='G. loss2')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.title('Train PSNR Progress')
plt.plot(NE_psnr_his1, label='U. PSNR1')
plt.plot(NE_psnr_his2, label='U. PSNR2')
plt.plot(IT_psnr_his1, label='G. PSNR1')
plt.plot(IT_psnr_his3, label='G. PSNR3')
plt.plot(DE_psnr_his, label='GainNet PSNR')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.title('Val PSNR Progress')
plt.plot(val_NE_psnr_his1, label='U. PSNR1')
plt.plot(val_NE_psnr_his2, label='U. PSNR2')
plt.plot(val_IT_psnr_his1, label='G. PSNR1')
plt.plot(val_IT_psnr_his3, label='G. PSNR2')
plt.plot(val_DE_psnr_his, label='GainNet PSNR')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save train, valid log
path2loss = './GainNet loss/'
os.makedirs(path2loss, exist_ok=True)
np.savetxt(os.path.join(path2loss, 'val_NE_loss_1st.txt'), val_NE_loss_his1)
np.savetxt(os.path.join(path2loss, 'val_NE_loss_2nd.txt'), val_NE_loss_his2)
np.savetxt(os.path.join(path2loss, 'val_gen_loss_1st.txt'), val_gen_loss_his1)
np.savetxt(os.path.join(path2loss, 'val_gen_loss_2nd.txt'), val_gen_loss_his3)
np.savetxt(os.path.join(path2loss, 'val_res_loss.txt'), val_DE_loss_his)

np.savetxt(os.path.join(path2loss, 'NE_loss_1st.txt'), NE_loss_his1)
np.savetxt(os.path.join(path2loss, 'NE_loss_2nd.txt'), NE_loss_his2)
np.savetxt(os.path.join(path2loss, 'gen_loss_1st.txt'), gen_loss_his1)
np.savetxt(os.path.join(path2loss, 'dis_loss_1st.txt'), dis_loss_his1)
np.savetxt(os.path.join(path2loss, 'gen_loss_2nd.txt'), gen_loss_his3)
np.savetxt(os.path.join(path2loss, 'dis_loss_2nd.txt'), dis_loss_his3)
np.savetxt(os.path.join(path2loss, 'res_loss.txt'), DE_loss_his)

np.savetxt(os.path.join(path2loss, 'NE_psnr_1st.txt'), NE_psnr_his1)
np.savetxt(os.path.join(path2loss, 'NE_psnr_2nd.txt'), NE_psnr_his2)
np.savetxt(os.path.join(path2loss, 'IT_psnr_1st.txt'), IT_psnr_his1)
np.savetxt(os.path.join(path2loss, 'IT_psnr_2nd.txt'), IT_psnr_his3)
np.savetxt(os.path.join(path2loss, 'res_psnr.txt'), DE_psnr_his)

np.savetxt(os.path.join(path2loss, 'val_NE_psnr_1st.txt'), val_NE_psnr_his1)
np.savetxt(os.path.join(path2loss, 'val_NE_psnr_2nd.txt'), val_NE_psnr_his2)
np.savetxt(os.path.join(path2loss, 'val_IT_psnr_1st.txt'), val_IT_psnr_his1)
np.savetxt(os.path.join(path2loss, 'val_IT_psnr_2nd.txt'), val_IT_psnr_his3)
np.savetxt(os.path.join(path2loss, 'val_res_psnr.txt'), val_DE_psnr_his)