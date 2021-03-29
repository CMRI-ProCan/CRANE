# Case study - Multicenter dataset

This case study shows how the multicenter dataset from Navarro, et al. (2016) [1] is de-noised via CRANE and it shows how the performance comparison results of Seneviratne et al. (2021) [2] were generated.

## Denoising via CRANE

The wiff files obtained from ProteomeXchange (Identifier—PXD002952) was first converted into the Toffee file format (Docker Image - cmriprocan/toffee:0.14.2) [3] and the PeakView compatible spectral libraries obtained from ProteomeXchange (Identifier—PXD002952) were converted using the OpenSwathLibraryFromPeakview class of OpenMSToffee (Docker Image - cmriprocan/openms-toffee:0.14.3). The raw Toffee files and the spectral libraries are available at ProteomeXchange (Identifier—PXD020529). The Raw toffee files needs to be saved in a sub directory called ``tof`` and the spectral libraries need to be saved in a subdirectory called ``srl`` for the de-noising scripts (crane_denoise_multicenter_*.py)  to work

e.g.

```
chmod +x ../setup-env.sh
bash ../setup-env.sh
conda activate crane
python crane_denoise_multicenter_32w_fix.py
```

## Performance analysis

Raw files and the CRANE denoised files were processed with and without the background subtraction inbuilt in OpenSWATH which we will call OSW0 and OSW1 respectively. Details of how the peptide data were generated is given in Seneviratne et al. (2021) [2]. multicenter_data_analysis.ipynb reads in the peptide data made available via ProteomeXchange (Identifier—PXD020529) and compares the performance of RAW_OSW0 with RAW_OSW1, CRANE_OSW0 and CRANE_OSW1 and generates the results given in Seneviratne et al. (2021) [2].

For the multicenter_data_analysis.ipynb notebook to work,

1. The peptide data should be saved in a subdirectory called ``peptide_n_protein_data``
2. diffacto version 1.0.5 needs to be copied to a subdirectory called ``diffacto`` (https://github.com/statisticalbiotechnology/diffacto)

multicenter_data_analysis.ipynb notebook generates the following outputs,

1. Figures used in Seneviratne et al. (2021) [2] in a subdirectory called ``figures``
2. All the statistics used in the tables of Seneviratne et al. (2021) [2] in a subdirectory called ``peptide_n_protein_data/stats``
3. All the input files requiered to run diffacto in a subdirectory called peptide_n_protein_data/diffacto_files

**Note:** If the protein data files made available at ProteomeXchange (Identifier—PXD020529) is saved in subdirectory called ``peptide_n_protein_data/diffacto_files`` user can skip the steps that run diffacto

## References

 1. P. Navarro, J. Kuharev, L. C. Gillet, O. M. Bernhardt, B. MacLean, H. L. Röst, S. A. Tate, C.-C. Tsou, L. Reiter, U. Distler, G. Rosenberger, Y. Perez-Riverol, A. I. Nesvizhskii, R. Aebersold and S. Tenzer, "A multicenter study benchmarks software tools for label-free proteome quantification." *Nature Biotechnology* 2016; 34: 1136.
 2. Akila J Seneviratne, Sean Peters, David Clarke, Michael Dausmann, Michael Hecker, Brett Tully, Peter G Hains and Qing Zhong, "Improved identification and quantification of peptides in mass spectrometry data via chemical and random additive noise elimination (CRANE)"
 3. Brett Tully, "Toffee – a highly efficient, lossless file format for DIA-MS". *Scientific Reports* 2020;10(1):8939