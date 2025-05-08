import random
import gc
import numpy as np
import sys
import torch
import datetime
from data_extraction import data_extraction
from modeling_data_load import modeling_data_loader
from train import model_training
from test import model_testing


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class Tee(object):
    def __init__(self, filepath):
        self.terminal = sys.__stdout__ 
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
set_seed(42)
data_path = "/home/fujiang/data_fusion/"
research_areas = ["1_SF", "2_SHIFT", "3_Kanab"]
arches = ["MSHFNET", "MSAHFNET"]

for area in research_areas:
    high_resolution = 5 if area == "2_SHIFT" else 4
    train_data, test_data, validate_data, scale_ratio, target_wvl, hr_wvl, \
            out_wvl, exclude_indices,test_hr_hsi_Geotrans, test_hr_hsi_proj = data_extraction(data_path, area, high_resolution)
    
    patch_size = 120
    n_bands = train_data[0].shape[2]
    n_epochs = 500
   
    print(f"***************************************************************************")
    for arch in arches:
        lr = 0.0001
        log_file = f"{data_path}2_saved_models/0_log_files/{area}_{arch}.log"
        sys.stdout = Tee(log_file)
        sys.stderr = sys.stdout
        print(f"Starting experiment: Area={area}, Model={arch}, Time={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        #****************************************************************************#
        
        data_loader, vali_list, test_list = modeling_data_loader(train_data, test_data, validate_data, patch_size, scale_ratio)
        model_training(arch, n_bands, n_epochs, lr, scale_ratio, target_wvl, hr_wvl, data_loader, data_path, area, vali_list)
        model_testing(arch, n_bands, scale_ratio, target_wvl, hr_wvl, out_wvl, 
                      exclude_indices, test_hr_hsi_Geotrans, test_hr_hsi_proj,
                      test_data, test_list, data_path, area)
        
        sys.stdout.log.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print(f"Finished: Area={area}, Model={arch}, Time={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        del data_loader, vali_list, test_list
        torch.cuda.empty_cache()
        gc.collect()
        
        

    