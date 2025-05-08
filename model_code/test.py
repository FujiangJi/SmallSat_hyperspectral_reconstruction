import math
import torch
import random
import numpy as np
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable
from models.MSAHFNET import MSHFNET, MSAHFNET
from data_preparation import array_to_geotiff
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def to_var(x, volatile=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device).float()
    return Variable(x, volatile=volatile)

def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

def model_testing(arch, n_bands, scale_ratio, target_wvl, hr_wvl, out_wvl, 
                  exclude_indices, test_hr_hsi_Geotrans, test_hr_hsi_proj,
                  test_data, test_list, data_path, area):
    set_seed(42)
    out_path = f"{data_path}2_saved_models/{area}/{arch}/"
    model_path = f"{out_path}{arch}_full_spectrum.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if arch == "MSHFNET":
        model = nn.DataParallel(MSHFNET(int(scale_ratio), 8, n_bands).cuda())
    elif arch == "MSAHFNET":
        model = nn.DataParallel(MSAHFNET(int(scale_ratio), 8, n_bands).cuda())

    checkpoint = torch.load(model_path)
    print(f"Best epoch: {checkpoint['epoch']}")
    model.load_state_dict(checkpoint['model_state_dict'])

    test_ref, test_lr, test_hr = test_list
    divisible_size = lcm(8, int(scale_ratio))
    h, w = test_ref.size(2), test_ref.size(3)
    new_h = (h // divisible_size) * divisible_size
    new_w = (w // divisible_size) * divisible_size
    test_ref = test_ref[:,:,0:new_h, 0:new_w]
    test_lr = test_lr[:,:,0:int(new_h/scale_ratio), 0:int(new_w/scale_ratio)]
    test_hr = test_hr[:,:,0:new_h, 0:new_w]
    print(test_ref.size(), test_lr.size(), test_hr.size())

    if area == "3_Kanab":
        image_size = min(new_h, new_w) 
        h_str = np.arange(0, h-image_size-1, image_size)
        w_str = np.arange(0, w-image_size-1, image_size)
        dataset = []
        for i in h_str:
            for j in w_str:
                test_ref_batch = test_ref[:,:, i:i+image_size, j:j+image_size]
                test_lr_batch = test_lr[:,:, int(i/scale_ratio):int((i+image_size)/scale_ratio), int(j/scale_ratio):int((j+image_size)/scale_ratio)]
                test_hr_batch = test_hr[:,:, i:i+image_size, j:j+image_size]
                test_batch = [test_ref_batch, test_lr_batch, test_hr_batch]
                dataset.append(test_batch)
        
        model.eval()
        with torch.no_grad():
            pred_list = []
            ref_list = []
            
            for idx, test_batch in enumerate(dataset):
                test_ref_batch, test_lr_batch, test_hr_batch = test_batch
                ref = to_var(test_ref_batch).detach()
                lr = to_var(test_lr_batch).detach()
                hr = to_var(test_hr_batch).detach()
                
                out = model(lr, hr)
                               
                pred_list.append(out.cpu())
                ref_list.append(ref.cpu())
            
            h, w = test_ref.shape[2], test_ref.shape[3]
            n_bands = test_ref.shape[1]

            out_full = torch.zeros((n_bands, h, w))
            ref_full = torch.zeros_like(out_full)
            count_map = torch.zeros((1, h, w))

            patch_idx = 0
            for i in h_str:
                for j in w_str:
                    out_patch = pred_list[patch_idx][0, :,:,:]
                    ref_patch = ref_list[patch_idx][0, :,:,:]

                    out_full[:, i:i+image_size, j:j+image_size] = out_patch
                    ref_full[:, i:i+image_size, j:j+image_size] = ref_patch
                    count_map[:, i:i+image_size, j:j+image_size] += 1

                    patch_idx += 1
            
            out_full /= count_map
            ref_full /= count_map

            out = out_full.numpy()
            ref = ref_full.numpy()

            valid_mask = ~np.isnan(ref)
            valid_mask = np.all(valid_mask, axis=0)
            rows = np.any(valid_mask, axis=1)
            cols = np.any(valid_mask, axis=0)
            row_start, row_end = np.where(rows)[0][[0, -1]]
            col_start, col_end = np.where(cols)[0][[0, -1]]

            ref = ref[:, row_start:row_end+1, col_start:col_end+1]
            out = out[:, row_start:row_end+1, col_start:col_end+1]
            out_im = np.moveaxis(out, 0, -1)
            
    else:
        model.eval()
        with torch.no_grad():
            ref = to_var(test_ref).detach()
            lr = to_var(test_lr).detach()
            hr = to_var(test_hr).detach()
            
            out = model(lr, hr)
            
            ref = ref.detach().cpu().numpy()
            out = out.detach().cpu().numpy()
            out_im = np.moveaxis(out[0,:,:,:], 0, -1)


    rmse = calc_rmse(ref, out)
    psnr = calc_psnr(ref, out)
    ergas = calc_ergas(ref, out)
    sam = calc_sam(ref, out)

    print(f"rmse: {rmse}, psnr:{psnr}:, ergas:{ergas}:, sam:{sam}")
    metric = pd.DataFrame([rmse, psnr, ergas, sam]).T
    metric.columns = ["RMSE", "PSNR","ERGAS","SAM"]
    metric["models"] = arch
    metric.to_csv(f"{out_path}/0_{arch}_metric.csv", index = False)


    final_out_im = np.full((out_im.shape[0], out_im.shape[1], len(out_wvl)), np.nan, dtype=np.float32)
    all_indices = np.arange(len(out_wvl))
    keep_indices = np.setdiff1d(all_indices, exclude_indices)

    final_out_im[:,:,keep_indices] = out_im
    out_tif = f"{out_path}{arch}_model_full_spectrum.tif"
    band_names = [f"{x} nm" for x in out_wvl]

    array_to_geotiff(out_im, out_tif, test_hr_hsi_Geotrans, test_hr_hsi_proj, band_names=band_names)
    out_tif = None
    out_im = None
    final_out_im = None
    return    
