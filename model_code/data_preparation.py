import os
import re
import numpy as np
import xarray as xr
from fwhm_resample import do_resample
import psutil
from joblib import Parallel, delayed
from osgeo import gdal, ogr,gdalconst

os.environ['PROJ_LIB'] = '/home/fujiang/miniconda3/envs/Fujiang/share/proj'
os.environ['GTIFF_SRS_SOURCE'] = 'EPSG'

def read_tif(tif_file):
    dataset = gdal.Open(tif_file)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    im_proj = (dataset.GetProjection())
    im_Geotrans = (dataset.GetGeoTransform())
    im_data = dataset.ReadAsArray(0, 0, cols, rows)
    if im_data.ndim == 3:
        im_data = np.moveaxis(dataset.ReadAsArray(0, 0, cols, rows), 0, -1)

    dataset = None
    return im_data, im_Geotrans, im_proj,rows, cols

def array_to_geotiff(array, output_path, geo_transform, projection, band_names=None):
    rows, cols, num_bands = array.shape
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, cols, rows, num_bands, gdal.GDT_Float32)
    
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    
    for band_num in range(num_bands):
        band = dataset.GetRasterBand(band_num + 1)
        band.WriteArray(array[:, :, band_num])
        band.FlushCache()
        
        if band_names:
            band.SetDescription(band_names[band_num])
    return
    
def get_shift_paras(head_file):
    with open(head_file, 'r') as file:
        hdr_content = file.read()
    wavelength_match = re.search(r'wavelength\s*=\s*\{(.*?)\}', hdr_content, re.DOTALL)
    if wavelength_match:
        wavelength_str = wavelength_match.group(1)
        hr_hsi_wvl = [float(value.strip()) for value in wavelength_str.split(',')]
    fwhm_match = re.search(r'fwhm\s*=\s*\{(.*?)\}', hdr_content, re.DOTALL)
    if fwhm_match:
        fwhm_str = fwhm_match.group(1)
        hr_hsi_fwhm = [float(value.strip()) for value in fwhm_str.split(',')]
    return hr_hsi_wvl, hr_hsi_fwhm

def get_emit_paras(head_file):
    ds_nc = xr.open_dataset(head_file, engine="h5netcdf", group="sensor_band_parameters")
    lr_wvl = ds_nc["wavelengths"].values
    lr_fwhm = ds_nc["fwhm"].values
    good_wvl = ds_nc["good_wavelengths"].values
    return lr_wvl, lr_fwhm, good_wvl

def split_image_chunks(chunk_size, image_array):
    image_chunks = []
    for i in range(0, image_array.shape[0], chunk_size):
        for j in range(0, image_array.shape[1], chunk_size):
            im_chunk = image_array[i:i + chunk_size, j:j + chunk_size]
            image_chunks.append(im_chunk)
    return image_chunks

def run_chunk_fwhm(image_chunk, source_wvl, target_wvl, source_fwhm):
    results = np.zeros(shape = (image_chunk.shape[0],image_chunk.shape[1],len(target_wvl)))
    for i in range(image_chunk.shape[0]):
        for j in range(image_chunk.shape[1]):
            spectra = image_chunk[i,j,:]
            spectra = np.nan_to_num(spectra, nan=0)
            spectra = do_resample(spectra, source_wvl, target_wvl, source_fwhm)
            results[i,j,:] = spectra
    return results

def do_image_fwhm(image_file, chunk_size,source_wvl,target_wvl,source_fwhm):
    image_array,im_Geotrans, im_proj,_,_= read_tif(image_file)
    image_chunks = split_image_chunks(chunk_size, image_array)
    num_processes = psutil.cpu_count(logical=False)
    chunk_results = Parallel(n_jobs=num_processes)(delayed(run_chunk_fwhm)(image_chunk, source_wvl, target_wvl, source_fwhm)
                                                   for image_chunk in image_chunks)
    
    final_array = np.zeros(shape = (image_array.shape[0], image_array.shape[1], len(target_wvl)))
    chunk_index = 0
    for i in range(0, final_array.shape[0], chunk_size):
        for j in range(0, final_array.shape[1], chunk_size):
            final_array[i:i + chunk_size, j:j + chunk_size,:] = chunk_results[chunk_index]
            chunk_index = chunk_index+1
    
    out_tif = f"{os.path.dirname(image_file)}/{os.path.basename(image_file)[:-4]}_fwhm.tif"
    band_names = [f"{x} nm" for x in target_wvl]
    array_to_geotiff(final_array, out_tif, im_Geotrans, im_proj, band_names=band_names)
    return

def resample_img(lr_para, hr_hsi_para, hr_file, chunk_size, target_wvl, low_resolution, high_resolution):
    lr_file, lr_wvl, lr_fwhm = lr_para
    hr_hsi_file, hr_hsi_wvl, hr_hsi_fwhm = hr_hsi_para

    """
    Apply FWHM spectral resampling to make sure the spectral resolution are same between lr_file and hr_hsi_file
    """
    print("start FWHM spectral resampling.....")
    do_image_fwhm(lr_file, chunk_size, lr_wvl, target_wvl, lr_fwhm) 
    do_image_fwhm(hr_hsi_file, chunk_size, hr_hsi_wvl, target_wvl, hr_hsi_fwhm)
    
    """
    Resample high resolution hyperspectral (hr_hsi) data.
    """
    print("start spatial resampling for hr_hsi.....")
    input_ds = gdal.Open(f"{os.path.dirname(hr_hsi_file)}/{os.path.basename(hr_hsi_file)[:-4]}_fwhm.tif")
    output_path = f"{os.path.dirname(hr_hsi_file)}/{os.path.basename(hr_hsi_file)[:-4]}_resampled.tif"
    gdal.Warp(output_path, input_ds, xRes=high_resolution, yRes=high_resolution,resampleAlg=gdalconst.GRA_Bilinear)
    input_df = None
    output_path = None

    """
    Resample low resolution (lr) data.
    """
    print("start spatial resampling for lr.....")
    input_ds = gdal.Open(f"{os.path.dirname(lr_file)}/{os.path.basename(lr_file)[:-4]}_fwhm.tif")
    output_path = f"{os.path.dirname(lr_file)}/{os.path.basename(lr_file)[:-4]}_resampled.tif"
    gdal.Warp(output_path, input_ds, xRes=low_resolution, yRes=low_resolution,resampleAlg=gdalconst.GRA_Bilinear)
    input_df = None
    output_path = None
        
    """
    Resample high resolution (hr) data.
    """
    print("start spatial resampling for hr.....")
    input_ds = gdal.Open(hr_file)
    output_path = f"{os.path.dirname(hr_file)}/{os.path.basename(hr_file)[:-4]}_resampled.tif"
    gdal.Warp(output_path, input_ds,xRes=high_resolution, yRes=high_resolution,resampleAlg=gdalconst.GRA_Bilinear)
    input_df = None
    output_path = None
    
    # print("extracting data.....")
    lr_data, lr_Geotrans, lr_proj,lr_rows, lr_cols = read_tif(f"{os.path.dirname(lr_file)}/{os.path.basename(lr_file)[:-4]}_resampled.tif")
    hr_data, hr_Geotrans, hr_proj,hr_rows, hr_cols = read_tif(f"{os.path.dirname(hr_file)}/{os.path.basename(hr_file)[:-4]}_resampled.tif")
    hr_hsi_data, hr_hsi_Geotrans, hr_hsi_proj,hr_hsi_rows, hr_hsi_cols = read_tif(f"{os.path.dirname(hr_hsi_file)}/{os.path.basename(hr_hsi_file)[:-4]}_resampled.tif")

    # print("deleting temp data.....")
    os.remove(f"{os.path.dirname(lr_file)}/{os.path.basename(lr_file)[:-4]}_fwhm.tif")
    os.remove(f"{os.path.dirname(lr_file)}/{os.path.basename(lr_file)[:-4]}_resampled.tif")
    os.remove(f"{os.path.dirname(hr_hsi_file)}/{os.path.basename(hr_hsi_file)[:-4]}_fwhm.tif")
    os.remove(f"{os.path.dirname(hr_hsi_file)}/{os.path.basename(hr_hsi_file)[:-4]}_resampled.tif")
    os.remove(f"{os.path.dirname(hr_file)}/{os.path.basename(lr_file)[:-4]}_resampled.tif")
    
    scale_ratio = lr_Geotrans[1]/hr_Geotrans[1]
    new_row = int(hr_data.shape[0] // scale_ratio * scale_ratio)
    new_col = int(hr_data.shape[1] // scale_ratio * scale_ratio)
    hr_img = hr_data[:new_row, :new_col, :]
    hr_hsi_img = hr_hsi_data[:new_row, :new_col, :]
    # row_edge = int(hr_data.shape[0]//scale_ratio*scale_ratio-hr_data.shape[0])
    # col_edge = int(hr_data.shape[1]//scale_ratio*scale_ratio-hr_data.shape[1])
    # row_edge = -1  if row_edge ==0  else row_edge
    # col_edge = -1  if col_edge ==0  else col_edge
    
    # hr_img = hr_data[:row_edge, :col_edge, :]
    # hr_hsi_img = hr_hsi_data[:hr_img.shape[0], :hr_img.shape[1], :]
    
    lr_rows = int(hr_img.shape[0]//scale_ratio)
    lr_cols = int(hr_img.shape[1]//scale_ratio)
    lr_img = lr_data[:lr_rows,:lr_cols,:]
    
    print(f"- Lower spatial resolution, hyperspectral imagery: {lr_img.shape} (rows, cols, bands) --> Geotransform: {lr_Geotrans}")
    print(f"- Higher spatial resolution, multispectral imagery: {hr_img.shape} (rows, cols, bands) --> Geotransform: {hr_Geotrans}")
    print(f"- Higher spatial resolution, hyperspectral imagery: {hr_hsi_img.shape} (rows, cols, bands) --> Geotransform: {hr_hsi_Geotrans}")
    
    return scale_ratio, lr_img,lr_Geotrans,lr_proj, hr_img, hr_Geotrans, hr_proj, hr_hsi_img, hr_hsi_Geotrans, hr_hsi_proj