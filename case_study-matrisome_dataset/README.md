# Case study - Matrisome dataset

This case study shows how the matrisome dataset from Krasny, et al. (2018) [1] is de-noised via CRANE and it shows how the performance comparison results of Seneviratne et al. (2021) [2] were generated.

## Denoising via CRANE

The wiff files obtained from ProteomeXchange (Identifier—PXD008651) was first converted into the Toffee file format (Docker Image - cmriprocan/toffee:0.14.2) [3]. 20171123_SWATH_Lu3_NoE_R1_1 raw file had two runs so the file had to be unpacked before converting to Toffee.

Krasny, et. al., (2018) specifically states Biognosys iRT peptides were spiked into samples for retention time alignment, so these were appended to the database (Sprot_mouse_20171020.fasta) used for searching. All Lung and Liver IDA runs (contained in files; 20170927_Spec_lib_Liver1.zip, 20171025_Spec_lib_Lung1.zip, 20171027_Spec_lib_Lung2.zip, 20171030_Spec_lib_Liver2.zip) were searched using ProteinPilot. The resultant group file was inspected and no Biognosys iRT peptides could be identified with an FDR <1%. As such, 14 endogenous peptides, common to all samples, were selected for retention time alignment using PeakView. The PeakView compatible spectral library was converted using the OpenSwathLibraryFromPeakview class of OpenMSToffee (Docker Image - cmriprocan/openms-toffee:0.14.3) using defaults parameters. 

The raw Toffee files and the spectral libraries are available at ProteomeXchange (Identifier—........). The Raw toffee files needs to be saved in a sub directory called ``tof`` and the spectral libraries need to be saved in a subdirectory called ``srl`` for the de-noising scripts (crane_denoise_matrisome.py)  to work

e.g.

```
chmod +x ../setup-env.sh
bash ../setup-env.sh
conda activate crane
python crane_denoise_matrisome.py
```

## Performance analysis

Raw files and the CRANE denoised files were processed with and without the background subtraction inbuilt in OpenSWATH which we will call OSW0 and OSW1 respectively. Details of how the peptide data were generated is given in Seneviratne et al. (2021) [2]. Matrisome_data_analysis.ipynb reads in the peptide data made available via ProteomeXchange (Identifier—........) and compares the performance of RAW_OSW0 with RAW_OSW1, CRANE_OSW0 and CRANE_OSW1 and generates the results given in Seneviratne et al. (2021) [2].

For the Matrisome_data_analysis.ipynb notebook to work,

1. The peptide data should be saved in a subdirectory called ``peptide_n_protein_data``
2. diffacto version 1.0.5 needs to be copied to a subdirectory called ``diffacto`` (https://github.com/statisticalbiotechnology/diffacto)

Matrisome_data_analysis.ipynb notebook generates the following outputs,

1. Figures used in Seneviratne et al. (2021) [2] in a subdirectory called ``figures``
2. All the statistics used in the tables of Seneviratne et al. (2021) [2] in a subdirectory called ``peptide_n_protein_data/stats``
3. All the input files requiered to run diffacto in a subdirectory called peptide_n_protein_data/diffacto_files

**Note:** If the protein data files made available at ProteomeXchange (Identifier—.........) is saved in subdirectory called ``peptide_n_protein_data/diffacto_files`` user can skip the steps that run diffacto

## References

 1. Krasny, L., et al. SWATH mass spectrometry as a tool for quantitative profiling of the matrisome. Journal of Proteomics 2018;189:11-22.
 2. Akila J Seneviratne, Sean Peters, David Clarke, Michael Dausmann, Michael Hecker, Brett Tully, Peter G Hains and Qing Zhong, "Improved identification and quantification of peptides in mass spectrometry data via chemical and random additive noise elimination (CRANE)"
 3. Brett Tully, "Toffee – a highly efficient, lossless file format for DIA-MS". *Scientific Reports* 2020;10(1):8939