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
from crane.denoiser import BEADS
from crane.mass_ranges import MassRangeCalculatorNonOverlappingPPMFullWidth
import pandas as pd

logging.basicConfig(filename='beads.log', level=logging.INFO)

tof_dir = 'tof/'

file_list = pd.read_csv('file_list.csv')
file_list = file_list.loc[~file_list.File_List.str.contains('_dda_')]
file_list = file_list.loc[file_list.File_List.str.contains('32fix')]
file_list = file_list.File_List.to_list()

irt_srl_path = 'srl/Navarro_openSwathIrt_32w_fix.tsv'
srl_path = 'srl/Navarro_openSwathDecoySrlTsv_32w_fix.tsv'

def create_BEADS_denoised_file(tof):
    ms1_denoiser = BEADS()
    ms2_denoiser = BEADS()
    tof_path = tof_dir + tof
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
                '.BEADS_32fix.tof'
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

for file in file_list:
    message = create_BEADS_denoised_file(file)
    print(message)