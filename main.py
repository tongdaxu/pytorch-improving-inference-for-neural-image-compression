import os
import torch
import torch.nn as nn
import torchvision 
from compressai.models import ScaleHyperprior
from compressai.zoo import load_state_dict
from dataset import KodakDataset
import torch.nn.functional as F
from net import ScaleHyperpriorSGA
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(prog='main')
parser.add_argument('-q', '--quality', required=True, help='quality = {0,...,7}')
parser.add_argument('-mr', '--model_root', required=True, help='root of model tar')
parser.add_argument('-dr', '--data_root', required=True, help='root of Kodak dataset')

def psnr(mse):
    return 10*torch.log10((255**2) / mse)

def main():
    args = parser.parse_args()
    model_root = args.model_root
    model_names = ["bmshj2018-hyperprior-1-7eb97409.pth.tar",
                   "bmshj2018-hyperprior-2-93677231.pth.tar",
                   "bmshj2018-hyperprior-3-6d87be32.pth.tar",
                   "bmshj2018-hyperprior-4-de1b779c.pth.tar",
                   "bmshj2018-hyperprior-5-f8b614e1.pth.tar",
                   "bmshj2018-hyperprior-6-1ab9c41e.pth.tar",
                   "bmshj2018-hyperprior-7-3804dcbd.pth.tar",
                   "bmshj2018-hyperprior-8-a583f0cf.pth.tar"]
    lams = [0.0018,0.0035,0.0067,0.0130,0.0250,0.0483,0.0932,0.1800]
    q = int(args.quality)
    Ns, Ms = [128,128,128,128,128,192,192,192], [192,192,192,192,192,320,320,320]
    N, M = Ns[q], Ms[q]
    
    model_path = os.path.join(model_root, model_names[q])
    model = ScaleHyperpriorSGA(N, M)
    model_dict = load_state_dict(torch.load(model_path))
    model.load_state_dict(model_dict)

    model = model.cuda()

    dataset = KodakDataset(kodak_root=args.data_root)
    dataloader = torch.utils.data.DataLoader(dataset)

    model.eval()
    bpp_init_avg, mse_init_avg, psnr_init_avg, rd_init_avg = 0, 0, 0, 0
    bpp_post_avg, mse_post_avg, psnr_post_avg, rd_post_avg = 0, 0, 0, 0

    tot_it = 2000
    lr = 5e-3
    for idx, img in enumerate(dataloader):
        img = img.cuda()
        img_h, img_w = img.shape[2], img.shape[3]
        img_pixnum = img_h * img_w
        # first round
        with torch.no_grad():
            ret_dict = model(img, "init")
        bpp_init = torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) / (img_pixnum) +\
                   torch.sum(-torch.log2(ret_dict["likelihoods"]["z"])) / (img_pixnum)
        mse_init = F.mse_loss(img, ret_dict["x_hat"]) * (255 ** 2)
        rd_init = bpp_init + lams[q] * mse_init
        psnr_init = psnr(mse_init)
        bpp_init_avg += bpp_init
        mse_init_avg += mse_init
        psnr_init_avg += psnr_init
        rd_init_avg += rd_init

        y, z = nn.parameter.Parameter(ret_dict["y"]), nn.parameter.Parameter(ret_dict["z"])
        opt = torch.optim.Adam([y] + [z], lr=lr)

        for it in range(tot_it):
            opt.zero_grad()   
            ret_dict = model(img, "sga", y, z, it, tot_it)
            bpp = torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) / (img_pixnum) + \
                  torch.sum(-torch.log2(ret_dict["likelihoods"]["z"])) / (img_pixnum)
            mse = F.mse_loss(img, ret_dict["x_hat"]) * (255 ** 2)
            rdcost = bpp + lams[q] * mse
            rdcost.backward()
            opt.step()

        with torch.no_grad():
            ret_dict = model(img, "round", y, z)

        bpp_post = torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) / (img_pixnum) +\
                   torch.sum(-torch.log2(ret_dict["likelihoods"]["z"])) / (img_pixnum)
        mse_post = F.mse_loss(img, ret_dict["x_hat"]) * (255 ** 2)
        rd_post = bpp_post + lams[q] * mse_post
        psnr_post = psnr(mse_post)
        bpp_post_avg += bpp_post
        mse_post_avg += mse_post
        psnr_post_avg += psnr_post
        rd_post_avg += rd_post

        print("img: {0}, psnr init: {1:.4f}, bpp init: {2:.4f}, rd init: {3:.4f}, psnr post: {4:.4f}, bpp post: {5:.4f}, rd post: {6:.4f}"\
              .format(idx, psnr_init, bpp_init, rd_init, psnr_post, bpp_post, rd_post))

    bpp_init_avg /= (idx + 1)
    mse_init_avg /= (idx + 1)
    psnr_init_avg /= (idx + 1)
    rd_init_avg /= (idx + 1)

    bpp_post_avg /= (idx + 1)
    mse_post_avg /= (idx + 1)
    psnr_post_avg /= (idx + 1)
    rd_post_avg /= (idx + 1)

    print("mean, psnr init: {0:.4f}, bpp init: {1:.4f}, rd init: {2:.4f}, psnr post: {3:.4f}, bpp post: {4:.4f}, rd post: {5:.4f}"\
          .format(psnr_init_avg, bpp_init_avg, rd_init_avg, psnr_post_avg, bpp_post_avg, rd_post_avg))

if __name__ == "__main__":
    main()