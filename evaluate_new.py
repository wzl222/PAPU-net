import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import transform
from scipy import ndimage


# -------------------------
# Reference-based metrics
# -------------------------
def getPSNR(img1, img2):
    return psnr(img1, img2, data_range=1.0)


def getSSIM(img1, img2):
    h, w = img1.shape[:2]
    if h < 7 or w < 7:
        return 0.0
    return ssim(img1, img2, data_range=1.0, channel_axis=-1, win_size=7)


# -------------------------
# No-reference metrics
# -------------------------
def getUCIQE(img):
    img = np.clip(img, 0, 1)
    img = (img * 255).astype('uint8')
    img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_LAB = np.array(img_LAB, dtype=np.float64)
    coe_Metric = [0.4680, 0.2745, 0.2576]
    img_lum = img_LAB[:, :, 0] / 255.0
    img_a = img_LAB[:, :, 1] / 255.0
    img_b = img_LAB[:, :, 2] / 255.0
    # item-1
    Img_Chr = np.sqrt(np.square(img_a) + np.square(img_b))
    Img_Sat = Img_Chr / np.sqrt(Img_Chr ** 2 + img_lum ** 2)
    Aver_Sat = np.mean(Img_Sat)
    Aver_Chr = np.mean(Img_Chr)
    Var_Chr = np.sqrt(np.mean((np.abs(1 - (Aver_Chr / (Img_Chr + 1e-12)) ** 2))))
    # item-2
    img_lum = img_lum.flatten()
    sorted_index = np.argsort(img_lum)
    top_index = sorted_index[int(len(img_lum) * 0.99)]
    bottom_index = sorted_index[int(len(img_lum) * 0.01)]
    con_lum = img_lum[top_index] - img_lum[bottom_index]
    uciqe = Var_Chr * coe_Metric[0] + con_lum * coe_Metric[1] + Aver_Sat * coe_Metric[2]
    return uciqe


# -------------------------
# UIQM sub-metrics (unchanged, only safe guards)
# -------------------------
def _uicm(img):
    img = np.array(img, dtype=np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    RG = R - G
    YB = (R + G) / 2 - B
    K = R.shape[0] * R.shape[1]
    RG1 = RG.reshape(1, K)
    RG1 = np.sort(RG1)
    alphaL = 0.1
    alphaR = 0.1
    RG1 = RG1[0, int(alphaL * K + 1):int(K * (1 - alphaR))]
    N = K * (1 - alphaR - alphaL)
    meanRG = np.sum(RG1) / N
    deltaRG = np.sqrt(np.sum((RG1 - meanRG) ** 2) / N)

    YB1 = YB.reshape(1, K)
    YB1 = np.sort(YB1)
    alphaL = 0.1
    alphaR = 0.1
    YB1 = YB1[0, int(alphaL * K + 1):int(K * (1 - alphaR))]
    N = K * (1 - alphaR - alphaL)
    meanYB = np.sum(YB1) / N
    deltaYB = np.sqrt(np.sum((YB1 - meanYB) ** 2) / N)
    uicm = -0.0268 * np.sqrt(meanRG ** 2 + meanYB ** 2) + 0.1586 * np.sqrt(deltaYB ** 2 + deltaRG ** 2)
    return uicm


def _uiconm(img):
    img = np.array(img, dtype=np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    patchez = 5
    m = R.shape[0]
    n = R.shape[1]
    if m % patchez != 0 or n % patchez != 0:
        x = int(m - m % patchez + patchez)
        y = int(n - n % patchez + patchez)
        R = transform.resize(R, (x, y), preserve_range=True, anti_aliasing=False)
        G = transform.resize(G, (x, y), preserve_range=True, anti_aliasing=False)
        B = transform.resize(B, (x, y), preserve_range=True, anti_aliasing=False)
    m = R.shape[0]
    n = R.shape[1]
    k1 = m / patchez
    k2 = n / patchez
    AMEER = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = R[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEER = AMEER + np.log((Max - Min) / (Max + Min)) * ((Max - Min) / (Max + Min))
    AMEER = 1 / (k1 * k2) * np.abs(AMEER)
    AMEEG = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = G[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEEG = AMEEG + np.log((Max - Min) / (Max + Min)) * ((Max - Min) / (Max + Min))
    AMEEG = 1 / (k1 * k2) * np.abs(AMEEG)
    AMEEB = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = B[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEEB = AMEEB + np.log((Max - Min) / (Max + Min)) * ((Max - Min) / (Max + Min))
    AMEEB = 1 / (k1 * k2) * np.abs(AMEEB)
    uiconm = AMEER + AMEEG + AMEEB
    return uiconm



def _uism(img):
    img = np.array(img, dtype=np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    hx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    hy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    SobelR = np.abs(ndimage.convolve(R, hx, mode='nearest') + ndimage.convolve(R, hy, mode='nearest'))
    SobelG = np.abs(ndimage.convolve(G, hx, mode='nearest') + ndimage.convolve(G, hy, mode='nearest'))
    SobelB = np.abs(ndimage.convolve(B, hx, mode='nearest') + ndimage.convolve(B, hy, mode='nearest'))
    patchez = 5
    m = R.shape[0]
    n = R.shape[1]
    if m % patchez != 0 or n % patchez != 0:
        x = int(m - m % patchez + patchez)
        y = int(n - n % patchez + patchez)
        SobelR = transform.resize(SobelR, (x, y), preserve_range=True, anti_aliasing=False)
        SobelG = transform.resize(SobelG, (x, y), preserve_range=True, anti_aliasing=False)
        SobelB = transform.resize(SobelB, (x, y), preserve_range=True, anti_aliasing=False)
    m = SobelR.shape[0]
    n = SobelR.shape[1]
    k1 = m / patchez
    k2 = n / patchez
    EMER = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = SobelR[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMER = EMER + np.log(Max / Min)
    EMER = 2 / (k1 * k2) * np.abs(EMER)

    EMEG = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = SobelG[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMEG = EMEG + np.log(Max / Min)
    EMEG = 2 / (k1 * k2) * np.abs(EMEG)
    EMEB = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = SobelB[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMEB = EMEB + np.log(Max / Min)
    EMEB = 2 / (k1 * k2) * np.abs(EMEB)
    lambdaR = 0.299
    lambdaG = 0.587
    lambdaB = 0.114
    uism = lambdaR * EMER + lambdaG * EMEG + lambdaB * EMEB
    return uism


def getUIQM(img):
    uicm = _uicm(img)
    uism = _uism(img)
    uiconm = _uiconm(img)
    return 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm, uicm, uism, uiconm


def _resize_to_patch_multiple_torch(img, patch=5):
    _, h, w = img.shape
    new_h = h if h % patch == 0 else h - h % patch + patch
    new_w = w if w % patch == 0 else w - w % patch + patch
    if new_h == h and new_w == w:
        return img
    return F.interpolate(
        img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)


def _patch_contrast_torch(channel, patch=5):
    patches = channel.unfold(0, patch, patch).unfold(1, patch, patch)
    patches = patches.contiguous().view(-1, patch * patch)
    max_v = patches.max(dim=1).values
    min_v = patches.min(dim=1).values
    mask = ((max_v != 0) | (min_v != 0)) & (max_v != min_v)
    if not torch.any(mask):
        return torch.zeros((), device=channel.device, dtype=channel.dtype)
    ratio = (max_v[mask] - min_v[mask]) / (max_v[mask] + min_v[mask] + 1e-12)
    return torch.abs(torch.sum(torch.log(ratio + 1e-12) * ratio) / patches.shape[0])


def _eme_torch(channel, patch=5):
    patches = channel.unfold(0, patch, patch).unfold(1, patch, patch)
    patches = patches.contiguous().view(-1, patch * patch)
    max_v = patches.max(dim=1).values
    min_v = patches.min(dim=1).values
    mask = (max_v != 0) & (min_v != 0)
    if not torch.any(mask):
        return torch.zeros((), device=channel.device, dtype=channel.dtype)
    return 2 * torch.abs(torch.sum(torch.log((max_v[mask] + 1e-12) / (min_v[mask] + 1e-12))) / patches.shape[0])


def getUIQM_torch(img, device):
    img = torch.from_numpy(np.asarray(img, dtype=np.float32)).to(device)
    img = img.permute(2, 0, 1)
    r, g, b = img[0], img[1], img[2]

    rg = (r - g).flatten().sort().values
    yb = ((r + g) / 2 - b).flatten().sort().values
    k = rg.numel()
    left = int(0.1 * k + 1)
    right = int(k * 0.9)
    rg_trim = rg[left:right]
    yb_trim = yb[left:right]
    mean_rg = rg_trim.mean()
    mean_yb = yb_trim.mean()
    delta_rg = torch.sqrt(torch.mean((rg_trim - mean_rg) ** 2))
    delta_yb = torch.sqrt(torch.mean((yb_trim - mean_yb) ** 2))
    uicm = -0.0268 * torch.sqrt(mean_rg ** 2 + mean_yb ** 2) + 0.1586 * torch.sqrt(delta_yb ** 2 + delta_rg ** 2)

    img_patch = _resize_to_patch_multiple_torch(img)
    uiconm = sum(_patch_contrast_torch(img_patch[c]) for c in range(3))

    hx = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device, dtype=img.dtype)
    hy = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=img.dtype)
    kernel = (hx + hy).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    sobel = torch.abs(F.conv2d(img.unsqueeze(0), kernel, padding=1, groups=3).squeeze(0))
    sobel = _resize_to_patch_multiple_torch(sobel)
    uism = 0.299 * _eme_torch(sobel[0]) + 0.587 * _eme_torch(sobel[1]) + 0.114 * _eme_torch(sobel[2])

    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    return uiqm.item(), uicm.item(), uism.item(), uiconm.item()


# -------------------------
# Main
# -------------------------
def is_image_file(name):
    return name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))


def read_rgb_image(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None
    rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_float = rgb_u8.astype(np.float32) / 255.0
    return rgb_u8, rgb_float


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--ref_dir', default=None)
    parser.add_argument('--save_txt', default='metrics.txt')
    parser.add_argument('--device', default='cuda', help='cuda, cuda:0, cuda:1, or cpu; GPU accelerates UIQM.')
    args = parser.parse_args()
    requested_device = args.device.lower()
    if requested_device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(requested_device)
        torch.cuda.set_device(device)
    else:
        if requested_device.startswith('cuda'):
            print('[Warn] CUDA is not available, falling back to CPU UIQM.')
        device = torch.device('cpu')
    use_gpu_uiqm = device.type == 'cuda'
    print(f'Using device for UIQM: {device}')

    input_names = sorted([f for f in os.listdir(args.input_dir) if is_image_file(f)])
    ref_names = sorted([f for f in os.listdir(args.ref_dir)]) if args.ref_dir else None

    cnt = 0
    sum_psnr = sum_ssim = sum_uiqm = sum_uicm = sum_uism = sum_uiconm = sum_uciqe = 0.0

    with open(args.save_txt, 'w') as f:
        for i, name in enumerate(input_names):
            img_path = os.path.join(args.input_dir, name)
            img_u8, img = read_rgb_image(img_path)

            if img_u8 is None:
                print(f"[Skip] Cannot read {name}")
                continue

            ref_img = None
            ref_img_u8 = None
            if args.ref_dir:
                ref_path = os.path.join(args.ref_dir, name)
                if not os.path.exists(ref_path):
                    print(f"[Skip] No reference for {name}")
                    continue
                ref_img_u8, ref_img = read_rgb_image(ref_path)
                if ref_img_u8 is None or img.shape != ref_img.shape:
                    print(f"[Skip] Shape mismatch: {name}")
                    continue

            if ref_img is not None:
                cur_psnr = getPSNR(img, ref_img)
                cur_ssim = getSSIM(img, ref_img)
            else:
                cur_psnr = cur_ssim = 0.0

            if use_gpu_uiqm:
                cur_uiqm, cur_uicm, cur_uism, cur_uiconm = getUIQM_torch(img_u8, device)
            else:
                cur_uiqm, cur_uicm, cur_uism, cur_uiconm = getUIQM(img_u8)
            cur_uciqe = getUCIQE(img)

            cnt += 1
            sum_psnr += cur_psnr
            sum_ssim += cur_ssim
            sum_uiqm += cur_uiqm
            sum_uicm += cur_uicm
            sum_uism += cur_uism
            sum_uiconm += cur_uiconm
            sum_uciqe += cur_uciqe

            print(f"Evaluated {name} ({i+1}/{len(input_names)})")
            f.write(
                f"{name} "
                f"PSNR={cur_psnr:.3f} SSIM={cur_ssim:.3f} "
                f"UIQM={cur_uiqm:.3f} UICM={cur_uicm:.3f} "
                f"UISM={cur_uism:.3f} UICONM={cur_uiconm:.3f} "
                f"UCIQE={cur_uciqe:.3f}\n"
            )

        if cnt > 0:
            f.write(
                f"\nAverage "
                f"PSNR={sum_psnr / cnt:.3f} SSIM={sum_ssim / cnt:.3f} "
                f"UIQM={sum_uiqm / cnt:.3f} UICM={sum_uicm / cnt:.3f} "
                f"UISM={sum_uism / cnt:.3f} UICONM={sum_uiconm / cnt:.3f} "
                f"UCIQE={sum_uciqe / cnt:.3f}\n"
            )

    print(f"Done. Valid images: {cnt}")
