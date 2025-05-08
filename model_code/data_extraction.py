
from data_preparation import resample_img, get_shift_paras, get_emit_paras
import numpy as np

def data_extraction(data_path, area, high_resolution):
    ### 1. Extract LR_HSI, HR_MSI, HR_HSI images from input imagery.
    head_file = f"{data_path}1_data/{area}/0_basic_information/HR_HSI_head.hdr"
    hr_hsi_wvl, hr_hsi_fwhm = get_shift_paras(head_file)
    head_file = f"{data_path}1_data/{area}/0_basic_information/LR_head.nc"
    lr_wvl, lr_fwhm, good_wvl = get_emit_paras(head_file)
    hr_wvl = [443, 490, 531, 565, 610, 665, 705, 865]

    """
    1.1 training images.
    """

    lr_file = f"{data_path}1_data/{area}/1_training_imagery/1_train_EMIT.tif"
    hr_file = f"{data_path}1_data/{area}/1_training_imagery/1_train_Planet.tif"
    hr_hsi = f"{data_path}1_data/{area}/1_training_imagery/1_train_HR_HSI.tif"

    lr_para, hr_hsi_para, hr_file  = [lr_file, lr_wvl, lr_fwhm], [hr_hsi, hr_hsi_wvl, hr_hsi_fwhm], hr_file
    chunk_size, target_wvl, low_resolution, high_resolution = 100, np.arange(400,2401,10), 60, high_resolution

    scale_ratio, train_lr_img,train_lr_Geotrans,train_lr_proj,\
    train_hr_img, train_hr_Geotrans, train_hr_proj, \
    train_hr_hsi_img, train_hr_hsi_Geotrans, train_hr_hsi_proj = resample_img(lr_para, hr_hsi_para, hr_file, chunk_size,
                                                                            target_wvl, low_resolution, high_resolution)

    """
    1.2 testing images.
    """

    lr_file = f"{data_path}1_data/{area}/2_testing_imagery/2_test_EMIT.tif"
    hr_file = f"{data_path}1_data/{area}/2_testing_imagery/2_test_Planet.tif"
    hr_hsi = f"{data_path}1_data/{area}/2_testing_imagery/2_test_HR_HSI.tif"

    lr_para, hr_hsi_para, hr_file  = [lr_file, lr_wvl, lr_fwhm], [hr_hsi, hr_hsi_wvl, hr_hsi_fwhm], hr_file
    chunk_size, target_wvl, low_resolution, high_resolution = 100, np.arange(400,2401,10), 60, high_resolution

    scale_ratio, test_lr_img,test_lr_Geotrans,test_lr_proj, \
    test_hr_img, test_hr_Geotrans, test_hr_proj, \
    test_hr_hsi_img, test_hr_hsi_Geotrans, test_hr_hsi_proj = resample_img(lr_para, hr_hsi_para, hr_file, chunk_size,
                                                                        target_wvl, low_resolution, high_resolution)

    """
    1.3 validating images.
    """

    lr_file = f"{data_path}1_data/{area}/3_validating_imagery/3_validate_EMIT.tif"
    hr_file = f"{data_path}1_data/{area}/3_validating_imagery/3_validate_Planet.tif"
    hr_hsi = f"{data_path}1_data/{area}/3_validating_imagery/3_validate_HR_HSI.tif"


    lr_para, hr_hsi_para, hr_file  = [lr_file, lr_wvl, lr_fwhm], [hr_hsi, hr_hsi_wvl, hr_hsi_fwhm], hr_file 
    chunk_size, target_wvl, low_resolution, high_resolution = 100, np.arange(400,2401,10), 60, high_resolution

    scale_ratio, vali_lr_img, vali_lr_Geotrans, vali_lr_proj, \
    vali_hr_img, vali_hr_Geotrans, vali_hr_proj, \
    vali_hr_hsi_img, vali_hr_hsi_Geotrans, vali_hr_hsi_proj = resample_img(lr_para, hr_hsi_para, hr_file, chunk_size, 
                                                                        target_wvl, low_resolution, high_resolution)


    ## exclude the bad bands
    bad_bands = [[1320, 1440], [1790, 2040]]
    exclude_indices = []
    for band_range in bad_bands:
        indices = np.where((target_wvl >= band_range[0]) & (target_wvl <= band_range[1]))[0]
        exclude_indices.extend(indices)

    exclude_indices = np.array(exclude_indices)

    out_wvl = target_wvl

    target_wvl = np.delete(target_wvl, exclude_indices)
    train_lr_img = np.delete(train_lr_img, exclude_indices, axis=-1)
    train_hr_hsi_img = np.delete(train_hr_hsi_img, exclude_indices, axis=-1)

    vali_lr_img = np.delete(vali_lr_img, exclude_indices, axis=-1)
    vali_hr_hsi_img = np.delete(vali_hr_hsi_img, exclude_indices, axis=-1)

    test_lr_img = np.delete(test_lr_img, exclude_indices, axis=-1)
    test_hr_hsi_img = np.delete(test_hr_hsi_img, exclude_indices, axis=-1)

    print("*******************************************************************************************")
    print(f"train lr image: {train_lr_img.shape}; train hr image: {train_hr_img.shape}; train hr hsi image: {train_hr_hsi_img.shape}")
    print(f"test lr image: {test_lr_img.shape}; test hr image: {test_hr_img.shape}; test hr hsi image: {test_hr_hsi_img.shape}")
    print(f"validation lr image: {vali_lr_img.shape}; validation hr image: {vali_hr_img.shape}; validation hr hsi image: {vali_hr_hsi_img.shape}")
    
    train_data = (train_hr_hsi_img, train_lr_img, train_hr_img)
    test_data = (test_hr_hsi_img, test_lr_img, test_hr_img)
    validate_data = (vali_hr_hsi_img, vali_lr_img, vali_hr_img)
    
    return train_data, test_data, validate_data, scale_ratio, target_wvl, hr_wvl, out_wvl, exclude_indices, test_hr_hsi_Geotrans, test_hr_hsi_proj