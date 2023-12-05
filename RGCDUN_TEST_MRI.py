
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
import glob
from time import time
import math
from torch.nn import init
import copy
import cv2
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim
from argparse import ArgumentParser
import types

parser = ArgumentParser(description='RGCDUN_MRI')

parser.add_argument('--epoch_num', type=int, default=410, help='epoch number of model')
parser.add_argument('--layer_num', type=int, default=8, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=50, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='save_temp', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='DataSets', help='training or test data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='BrainImages_test', help='name of test set')

args = parser.parse_args()


epoch_num = args.epoch_num
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
test_name = args.test_name


try:
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
    torch.backends.cuda.matmul.allow_tf32 = False
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = False
except:
    pass


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/mask_%d.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
mask_matrix = Phi_data['mask_matrix']


mask_matrix = torch.from_numpy(mask_matrix).type(torch.FloatTensor)
mask = torch.unsqueeze(mask_matrix, 2)
mask = torch.cat([mask, mask], 2)
mask = mask.to(device)


if isinstance(torch.fft, types.ModuleType):
    class FFT_Mask_ForBack(torch.nn.Module):
        def __init__(self):
            super(FFT_Mask_ForBack, self).__init__()

        def forward(self, x, full_mask):
            full_mask = full_mask[..., 0]
            x_in_k_space = torch.fft.fft2(x)
            masked_x_in_k_space = x_in_k_space * full_mask.view(1, 1, *(full_mask.shape))
            masked_x = torch.real(torch.fft.ifft2(masked_x_in_k_space))
            return masked_x
else:
    class FFT_Mask_ForBack(torch.nn.Module):
        def __init__(self):
            super(FFT_Mask_ForBack, self).__init__()

        def forward(self, x, mask):
            x_dim_0 = x.shape[0]
            x_dim_1 = x.shape[1]
            x_dim_2 = x.shape[2]
            x_dim_3 = x.shape[3]
            x = x.view(-1, x_dim_2, x_dim_3, 1)
            y = torch.zeros_like(x)
            z = torch.cat([x, y], 3)
            fftz = torch.fft(z, 2)
            z_hat = torch.ifft(fftz * mask, 2)
            x = z_hat[:, :, :, 0:1]
            x = x.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)
            return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class MSIM(nn.Module):
    def __init__(self, dim):
        super(MSIM, self).__init__()
        self.cat = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, stride=1)
        self.grad = GradBlock(dim)
        # base
        self.conv1 = nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, stride=1, bias=True)
        self.res1 = ResBlock(dim * 2)
        self.res2 = ResBlock(dim * 2)
        self.conv2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, stride=1, bias=True)
        # lstm
        pad_x = 1
        self.conv_xf = nn.Conv2d(dim, dim, kernel_size=3, padding=pad_x)
        self.conv_xi = nn.Conv2d(dim, dim, kernel_size=3, padding=pad_x)
        self.conv_xo = nn.Conv2d(dim, dim, kernel_size=3, padding=pad_x)
        self.conv_xj = nn.Conv2d(dim, dim, kernel_size=3, padding=pad_x)

        pad_h = 1
        self.conv_hf = nn.Conv2d(dim, dim, kernel_size=3, padding=pad_h)
        self.conv_hi = nn.Conv2d(dim, dim, kernel_size=3, padding=pad_h)
        self.conv_ho = nn.Conv2d(dim, dim, kernel_size=3, padding=pad_h)
        self.conv_hj = nn.Conv2d(dim, dim, kernel_size=3, padding=pad_h)

    def forward(self, x, h, c, PhiTb,mask):
        if h is None and c is None:

            i = F.sigmoid(self.conv_xi(x))
            o = F.sigmoid(self.conv_xo(x))
            j = F.tanh(self.conv_xj(x))
            x = self.grad(x, PhiTb,mask)
            c = x + self.conv2(self.res2(self.res1(self.conv1(self.cat(torch.cat([x, i * j], dim=1))))))
            h = o * F.tanh(c)

        else:

            f = F.sigmoid(self.conv_xf(x) + self.conv_hf(h))
            i = F.sigmoid(self.conv_xi(x) + self.conv_hi(h))
            o = F.sigmoid(self.conv_xo(x) + self.conv_ho(h))
            j = F.tanh(self.conv_xj(x) + self.conv_hj(h))
            c = f * c
            c = self.grad(c, PhiTb,mask)
            c = c + self.conv2(self.res2(self.res1(self.conv1(self.cat(torch.cat([c, i * j], dim=1))))))
            h = o * F.tanh(c)

        return c, h, c


class GradBlock(nn.Module):
    def __init__(self, dim):
        super(GradBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=True)
        self.res1 = ResBlock(dim)
        self.fft_forback = FFT_Mask_ForBack()
    def forward(self, x,  PhiTb,mask):
        x_pixel = self.conv1(x)
        PhiTPhix = self.fft_forback(x_pixel,mask)
        PhiTPhix_PhiTb = PhiTPhix-PhiTb
        x_delta = self.conv2(PhiTPhix_PhiTb)
        x = self.res1(x_delta) + x
        return x


class RMSDUN(nn.Module):
    def __init__(self,LayerNo):
        super(RMSDUN, self).__init__()

        self.base = 16
        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(self.base, 1, kernel_size=1, padding=0, bias=True)
        layer1 = []
        self.LayerNo = LayerNo
        for i in range(LayerNo):
            layer1.append(MSIM(self.base))
        self.fcs1 = nn.ModuleList(layer1)

    def forward(self, PhiTb,mask):

        x = PhiTb
        x = self.conv1(x)
        h = None
        c = None
        for i in range(self.LayerNo):
            if i == 0:
                c, h, x = self.fcs1[i](x, h, c, PhiTb,mask)
            elif i == self.LayerNo - 1:
                c, h, x = self.fcs1[i](x, h, c,  PhiTb,mask)
            else:
                for j in range(5):
                    c, h, x = self.fcs1[i](x, h, c,  PhiTb,mask)
        x = self.conv2(c)
        return x



model = RMSDUN(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

print_flag = 1   # print parameter number
num_count = 0
num_params = 0
for para in model.parameters():
    num_count += 1
    num_params += para.numel()
print("total para num: %d" % num_params)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "%s/MRI_RGCDUN_layer_%d_group_%d_ratio_%d" % (args.model_dir, layer_num, group_num, cs_ratio)

# Load pre-trained model with epoch number
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_num)))


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


test_dir = os.path.join(args.data_dir, test_name)
filepaths = glob.glob(test_dir + '/*.png')

result_dir = os.path.join(args.result_dir, test_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

Init_PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
Init_SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)


print('\n')
print("MRI CS Reconstruction Start")

with torch.no_grad():
    for img_no in range(ImgNum):

        imgName = filepaths[img_no]

        Iorg = cv2.imread(imgName, 0)

        Icol = Iorg.reshape(1, 1, 256, 256) / 255.0

        Img_output = Icol

        start = time()

        batch_x = torch.from_numpy(Img_output)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)

        PhiTb = FFT_Mask_ForBack()(batch_x, mask)

        x_output = model(PhiTb, mask)

        end = time()

        initial_result = PhiTb.cpu().data.numpy().reshape(256, 256)
        Prediction_value = x_output.cpu().data.numpy().reshape(256, 256)


        X_init = np.clip(initial_result, 0, 1).astype(np.float64)
        X_rec = np.clip(Prediction_value, 0, 1).astype(np.float64)

        init_PSNR = psnr(X_init * 255, Iorg.astype(np.float64))
        init_SSIM = ssim(X_init * 255, Iorg.astype(np.float64), data_range=255)

        rec_PSNR = psnr(X_rec*255., Iorg.astype(np.float64))
        rec_SSIM = ssim(X_rec*255., Iorg.astype(np.float64), data_range=255)

        Img_output = Img_output.reshape(256,256)
        x_output= abs(Img_output-Prediction_value)
        # x_max = np.max(x_output)
        # x_min = np.min(x_output)
        # x_output = (x_output-x_min)/(x_max-x_min)
        x_output = x_output*255
        # x_output = cv2.applyColorMap(x_output, cv2.COLORMAP_JET)
        print("[%02d/%02d] Run time for %s is %.4f, Initial  PSNR is %.2f, Initial  SSIM is %.4f" % (img_no, ImgNum, imgName, (end - start), init_PSNR, init_SSIM))
        print("[%02d/%02d] Run time for %s is %.4f, Proposed PSNR is %.2f, Proposed SSIM is %.4f" % (img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM))

        im_rec_rgb = np.clip(X_rec*255, 0, 255).astype(np.uint8)

        resultName = imgName.replace(args.data_dir, args.result_dir)

        cv2.imwrite("%s_errormap_32_ratio_%d_epoch_%d_PSNR_%.2f_SSIM_%.4f.bmp" % (
        resultName, cs_ratio, epoch_num, rec_PSNR, rec_SSIM), x_output)
        cv2.imwrite("%s_HTDIDUN_32_ratio_%d_epoch_%d_PSNR_%.2f_SSIM_%.4f.bmp" % (resultName, cs_ratio, epoch_num, rec_PSNR, rec_SSIM), im_rec_rgb)
        del x_output

        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM

        Init_PSNR_All[0, img_no] = init_PSNR
        Init_SSIM_All[0, img_no] = init_SSIM

print('\n')
init_data =   "CS ratio is %d, Avg Initial  PSNR/SSIM for %s is %.2f/%.4f" % (cs_ratio, args.test_name, np.mean(Init_PSNR_All), np.mean(Init_SSIM_All))
output_data = "CS ratio is %d, Avg Proposed PSNR/SSIM for %s is %.2f/%.4f, Epoch number of model is %d \n" % (cs_ratio, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), epoch_num)
print(init_data)
print(output_data)
print("MRI CS Reconstruction End")