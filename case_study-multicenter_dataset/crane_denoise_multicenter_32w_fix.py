# ======================================================================
# In this script a string of digits is used to specify the tuning parameters 
# used to produce each denoised file. Meaning of the digits are as follows,
# Digit 0 : Artifact filter and Inverse transform setting
#          0 = Artefact filter with raw data and max_denoise = False
#          1 = Artefact filter with thresholded data and max_denoise = False
#          2 = Artefact filter with raw data and max_denoise = True
# Digit 1 : Signal_preconditioning Transform
#          0 = No transform
#          1 = log
#          2 = Anscombe
# Digit 2 : Split data, 1 = True, 0 = False
# Digit 3 : Wavelet nshape, 0 = db2, 1 = haar, 2 = db3, 3 = coif1, 4 = coif2
# Digit 4 : Apply HRMC, 1 = True, 0 = False
# Digit 5 : Thresholding, 
#           0 = VisuShrink, 
#           1 = BayesShrink, 
#           2 = Nothresholding,
#           3 = SUREShrink
# Digit 6 : Number of levels of smooth coefficient suppression
# Digit 7 : Number of levels of wavelet transform
# e.g. param_set = 101134 means
#      - Split data into small squares before denoising
#      - Use db2 wavelet
#      - Apply Horizontal Row Median Correction 
#      - Use BayesShrink to calculate the threshold
#      - Suppress the first three levels of smooth coefficients
#      - Use four levels of wavelet decomposition
# ======================================================================
#
# The Raw toffee files needs to be saved in a sub directory called tof 
# and the srls need to be saved in a subdirectory called srl for the script 
# to work
# 
# ======================================================================
import os
import sys
import logging

# Add path to the directory where crane folder is stored
path = os.path.realpath(__file__)
dirname = os.path.dirname(path)
sys.path.append(os.path.join(dirname, '..'))
# Import crane classes
from crane.app import App
from crane.denoiser import Crane, Thresholding, Transform, DenoiserNoOp
from crane.mass_ranges import MassRangeCalculatorNonOverlappingPPMFullWidth
import pandas as pd

logging.basicConfig(filename='crane.log', level=logging.INFO)

tof_dir = 'tof/'

file_list = pd.read_csv('file_list.csv')
file_list = file_list.loc[~file_list.File_List.str.contains('_dda_')]
file_list = file_list.loc[file_list.File_List.str.contains('32fix')]
file_list = file_list.File_List.to_list()

irt_srl_path = 'srl/Navarro_openSwathIrt_32w_fix.tsv'
srl_path = 'srl/Navarro_openSwathDecoySrlTsv_32w_fix.tsv'

MS1_settings_list = [
    '00001066',
]
MS2_settings_list = [
]

def initialise_denoiser_crane(param_set):
    artifact_filter_with_thresholded_XIC = False
    max_denoise = False
    if int(param_set[0]) == 1:
        artifact_filter_with_thresholded_XIC = True
    if int(param_set[0]) == 2:
        max_denoise = True
    
    transform_technique=Transform.NoTransform
    if int(param_set[1]) == 1:
        transform_technique=Transform.Log
    if int(param_set[1]) == 2:
        transform_technique=Transform.Anscombe1
    if int(param_set[1]) == 3:
        transform_technique=Transform.Anscombe2
    if int(param_set[1]) == 4:
        transform_technique=Transform.Anscombe3
    
    split_data = False
    if int(param_set[2]) == 1:
        split_data = True
    
    wavelet_method='db2'
    if int(param_set[3]) == 1:
        wavelet_method='haar'
    if int(param_set[3]) == 2:
        wavelet_method='db3'
    if int(param_set[3]) == 3:
        wavelet_method='coif1'
    if int(param_set[3]) == 4:
        wavelet_method='coif2'
    
    apply_hrmc = False
    if int(param_set[4]) == 1:
        apply_hrmc = True
    
    thresholding_technique = Thresholding.VisuShrink
    if int(param_set[5]) == 1:
        thresholding_technique = Thresholding.BayesShrink
    if int(param_set[5]) == 2:
        thresholding_technique = Thresholding.NoThresholding
    if int(param_set[5]) == 3:
        thresholding_technique = Thresholding.SUREShrink
    
    smooth_coeff_suppression = int(param_set[6])
    
    wt_levels = int(param_set[7])
    
    denoiser = Crane(
        transform_technique=transform_technique,
        wavelet_method=wavelet_method,
        levels=wt_levels,
        apply_hrmc=apply_hrmc,
        split_data=split_data,
        thresholding_technique=thresholding_technique,
        smooth_coeff_suppression=smooth_coeff_suppression,
        artifact_filter_with_thresholded_XIC=artifact_filter_with_thresholded_XIC,
        max_denoise=max_denoise,
    )
    return denoiser

def crane_dilution_with_param_set(tof, param_set1, param_set2):
    ms1_denoiser = initialise_denoiser_crane(param_set1)
    ms2_denoiser = initialise_denoiser_crane(param_set2)
    tof_path = tof_dir + tof

    if ms1_denoiser.levels > 7:
        split_size = ms1_denoiser.levels ** 2
    else:
        split_size = 128

    if len(tof_path) > 0:
        denoiser = App(
            tof_fname=tof_path,
            srl_paths=[srl_path, irt_srl_path],
            denoiser=ms2_denoiser,
            mass_range_calculator=MassRangeCalculatorNonOverlappingPPMFullWidth(
                ppm_full_width=100,
                split_size=split_size,
            ),
            ms1_denoiser=ms1_denoiser,  # None = same as MS2
            denoised_fname=tof_path.replace(
                '.tof',
                '.crane_MS1_{}_MS2_{}_32fix.tof'.format(param_set1, param_set2)
            ),
            window_name=None,
            include_ms1=True,
        )
        if not os.path.isfile(denoiser.denoised_fname):
            print('Denoising:', denoiser.denoised_fname)
            denoiser.run()
            return denoiser.denoised_fname + ' created successfully'
        else:
            return denoiser.denoised_fname + ' already exists'

for param_set1 in MS1_settings_list:
#     for param_set2 in MS2_settings_list:
    param_set2 = param_set1
    for file in file_list:
        message = crane_dilution_with_param_set(file, param_set1, param_set2)
        print(message)