import torch
import random
import numpy as np
from torch.utils.data import DataLoader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def modeling_data_loader(train_data, test_data, validate_data, image_size, scale_ratio):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Use {device} to train models")

    train_ref, train_lr, train_hr = train_data
    test_ref, test_lr, test_hr = test_data
    vali_ref, vali_lr, vali_hr = validate_data

    train_ref = torch.from_numpy(train_ref).permute(2,0,1).float().to(device)
    train_lr = torch.from_numpy(train_lr).permute(2,0,1).float().to(device) 
    train_hr = torch.from_numpy(train_hr).permute(2,0,1).float().to(device) 
    
    test_ref = torch.from_numpy(test_ref).permute(2,0,1).unsqueeze(dim=0).float().to(device)
    test_lr = torch.from_numpy(test_lr).permute(2,0,1).unsqueeze(dim=0).float().to(device)
    test_hr = torch.from_numpy(test_hr).permute(2,0,1).unsqueeze(dim=0).float().to(device)

    vali_ref = torch.from_numpy(vali_ref).permute(2,0,1).unsqueeze(dim=0).float().to(device)
    vali_lr = torch.from_numpy(vali_lr).permute(2,0,1).unsqueeze(dim=0).float().to(device)
    vali_hr = torch.from_numpy(vali_hr).permute(2,0,1).unsqueeze(dim=0).float().to(device)

    print(f"train size, ref: {train_ref.shape}, lr: {train_lr.shape}, hr: {train_hr.shape}")
    print(f"test size, ref: {test_ref.shape}, lr: {test_lr.shape}, hr:{test_hr.shape}")
    print(f"validation size, ref: {vali_ref.shape}, lr: {vali_lr.shape}, hr:{vali_hr.shape}")

    train_list = [train_ref, train_lr, train_hr]
    vali_list = [vali_ref, vali_lr, vali_hr]
    test_list = [test_ref, test_lr, test_hr]

    train_ref, train_lr, train_hr = train_list
    h, w = train_ref.size(1), train_ref.size(2)
    print(h,w)

    h_str = np.arange(0, h-image_size-1, image_size*0.9).astype(int)
    w_str = np.arange(0, w-image_size-1, image_size*0.9).astype(int)

    dataset = []
    for i in h_str:
        for j in w_str:
            train_ref_batch = train_ref[:, i:i+image_size, j:j+image_size]
            train_lr_batch = train_lr[:, int(i/scale_ratio):int((i+image_size)/scale_ratio), int(j/scale_ratio):int((j+image_size)/scale_ratio)]
            train_hr_batch = train_hr[:, i:i+image_size, j:j+image_size]
            train_batch = [train_ref_batch, train_lr_batch, train_hr_batch]
            dataset.append(train_batch)
            
    g = torch.Generator()
    g.manual_seed(42)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True,worker_init_fn=seed_worker,generator=g)
    return data_loader, vali_list, test_list