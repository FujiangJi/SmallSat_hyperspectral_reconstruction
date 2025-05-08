import os
import gc
import math
import torch
import random
import datetime
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from models.MSAHFNET import MSHFNET, MSAHFNET
from models.loss import FusionLoss

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

def model_training(arch, n_bands, n_epochs, lr, scale_ratio, target_wvl, hr_wvl, data_loader, data_path, area, vali_list):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_path = f"{data_path}2_saved_models/{area}/"
    os.makedirs(f"{out_path}{arch}", exist_ok=True)
    out_path = f"{out_path}{arch}/"
    print(f"output path: {out_path}")
    
    if arch == "MSHFNET":
        model = nn.DataParallel(MSHFNET(int(scale_ratio), 8, n_bands).cuda())
    elif arch == "MSAHFNET":
        model = nn.DataParallel(MSAHFNET(int(scale_ratio), 8, n_bands).cuda())

        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    parameter_nums = sum(p.numel() for p in model.parameters())
    print("Model size:", str(float(parameter_nums / 1e6)) + 'M')
    model_path = f"{out_path}{arch}_full_spectrum.pth"

    criterion = FusionLoss().to(device) if arch == "MSAHFNET" else nn.MSELoss().to(device)
    
    start_t = datetime.datetime.now()
    print(f'{start_t}: Start training {arch} model')

    loss_all = []
    loss_vali_all = []
    best_val_loss = float("inf")
    for epoch in range(n_epochs):
        model.train()
        loss_sum = 0
        loss_vali_sum = 0
        for train_batch in data_loader:
            train_ref_batch, train_lr_batch, train_hr_batch = train_batch
            
            image_lr = to_var(train_lr_batch).detach()
            image_hr = to_var(train_hr_batch).detach()
            image_ref = to_var(train_ref_batch).detach()
            
            # Forward, Backward and Optimize
            optimizer.zero_grad()
            
            out = model(image_lr, image_hr)
            loss = criterion(out, image_ref)
            
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()

            ### model validation
            vali_ref, vali_lr, vali_hr = vali_list
            divisible_size = lcm(8, int(scale_ratio))
            h, w = vali_ref.size(2), vali_ref.size(3)
            new_h = (h // divisible_size) * divisible_size
            new_w = (w // divisible_size) * divisible_size
            vali_ref = vali_ref[:,:,0:new_h, 0:new_w]
            vali_lr = vali_lr[:,:,0:int(new_h/scale_ratio), 0:int(new_w/scale_ratio)]
            vali_hr = vali_hr[:,:,0:new_h, 0:new_w]
            
            if area == "3_Kanab":
                image_size = min(new_h, new_w) 
                h_str = np.arange(0, h-image_size-1, image_size)
                w_str = np.arange(0, w-image_size-1, image_size)
                dataset = []
                for i in h_str:
                    for j in w_str:
                        vali_ref_batch = vali_ref[:,:, i:i+image_size, j:j+image_size]
                        vali_lr_batch = vali_lr[:,:, int(i/scale_ratio):int((i+image_size)/scale_ratio), int(j/scale_ratio):int((j+image_size)/scale_ratio)]
                        vali_hr_batch = vali_hr[:,:, i:i+image_size, j:j+image_size]
                        vali_batch = [vali_ref_batch, vali_lr_batch, vali_hr_batch]
                        dataset.append(vali_batch)
                
                model.eval()
                with torch.no_grad():
                    loss_vali_batch = []
                    for vali_batch in dataset:
                        vali_ref_batch, vali_lr_batch, vali_hr_batch = vali_batch
                        ref_vali = to_var(vali_ref_batch).detach()
                        lr_vali = to_var(vali_lr_batch).detach()
                        hr_vali = to_var(vali_hr_batch).detach()
                        
                        out_vali = model(lr_vali, hr_vali)
                        
                        loss_vali = criterion(ref_vali, out_vali)
                        loss_vali_batch.append(loss_vali.item())
                    temp = np.mean(loss_vali_batch)
            
            else:
                model.eval()
                with torch.no_grad():
                    ref_vali = to_var(vali_ref).detach()
                    lr_vali = to_var(vali_lr).detach()
                    hr_vali = to_var(vali_hr).detach()
                    
                    out_vali = model(lr_vali, hr_vali)

                    loss_vali = criterion(ref_vali, out_vali)
                    temp = loss_vali.item()    
                    
            loss_vali_sum += temp 
        
            del train_ref_batch, train_lr_batch, train_hr_batch
            del image_lr, image_hr, image_ref, out, loss
            del ref_vali, lr_vali, hr_vali, out_vali, loss_vali
            gc.collect()
            model.train()
            
        loss_all.append(loss_sum)
        loss_vali_all.append(loss_vali_sum)

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'Epoch [%d/%d], train_Loss: %.4f' % (epoch, n_epochs, loss_sum))
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'Epoch [%d/%d], validation_Loss: %.4f' % (epoch, n_epochs, loss_vali_sum))
        if loss_vali_sum < best_val_loss:
            best_val_loss = loss_vali_sum
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss}, model_path)          
        torch.cuda.empty_cache()
        gc.collect()

    loss_all = pd.DataFrame(np.array(loss_all), columns = ["train_loss"])
    loss_vali_all = pd.DataFrame(np.array(loss_vali_all), columns = ["validation_loss"])
    loss_train_vali_all = pd.concat([loss_all, loss_vali_all],axis = 1)
    loss_train_vali_all["epochs"] = np.arange(1,n_epochs+1)
    loss_train_vali_all.to_csv(f"{out_path}/0_{arch}_loss.csv", index = False)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,3))
    plt.subplots_adjust(wspace = 0.25)
    ax1.set_facecolor((0,0,0,0.02))
    ax1.grid(color='gray', linestyle=':', linewidth=0.3)
    ax2.set_facecolor((0,0,0,0.02))
    ax2.grid(color='gray', linestyle=':', linewidth=0.3) 
        
    ax1.plot(loss_train_vali_all["epochs"], loss_train_vali_all["train_loss"], color = "red", lw = 2)
    ax2.plot(loss_train_vali_all["epochs"], loss_train_vali_all["validation_loss"], color = "blue", lw = 2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss (MSE)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Validation Loss (MSE)')
    plt.savefig(f"{out_path}/0_{arch}_epoch_full_spectrum.png", dpi=200, bbox_inches='tight')
    
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print(f'{end_t}: Complete training {arch} model')
    print(f'Total time consumption for train {arch} model: {elapsed_sec/3600} hours')
    return