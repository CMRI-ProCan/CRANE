# Chemical and Random Additive Noise Elimination (CRANE)

CRANE is a wavelet transform based method for de-noising electrospray ionisation - liquid chromatography mass spectrometry data [1]

## Set up environment

Inorder to set up the environment, conda is a prerequisite (https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
In the command line run setup-env.sh

```
bash setup-env.sh
conda activate crane
```

## Set up path 

path to the directory where crane is stored should be setup in the python code before being able to import the CRANE classes.

`` sys.path.append(crane_path) ``

where ``crane_path`` is the absolute path to the directory where crane is stored. See the crane_denoise_multicenter_*.py scripts in the case study section for an example.
 
## Usage

Current implementation of CRANE accepts mass spectrometry data files in the toffee file format (Docker Image - cmriprocan/toffee:0.14.2) [2] as input. Currently the toffee file format only support Time-Of-Flight (TOF), Data-Independent Acquisition (DIA) data. However the general approach of CRANE de-noising would support all forms of electrospray ionisation - liquid chromatography mass spectrometry data.

 - ``App`` class of ``crane.app`` manages the denoising of the MS file and creating a new denoised data file. 
 - ``App`` accepts openSWATH compatible SRLs. Alternatively one could convert PeakView compatible SRLs using the OpenSwathLibraryFromPeakview class of OpenMSToffee (Docker Image - cmriprocan/openms-toffee:0.14.3)
 - Since vast amount of computer memory is required to denoise an entire MS window at once, strips of data are extracted from the MS window and processed independently. Therefore a suitable instance of the ``MassRangeCalculatorBase`` class of ``crane.mass_ranges`` should be passed when initializing ``crane.app``
     - If the entire window is to be denoised use ```MassRangeCalculatorFullMapSplit``` class
     - If only the areas looked at by OpenSWATH is to be denoised use an instance of either ```MassRangeCalculatorNonOverlappingPPMFullWidth``` or ```MassRangeCalculatorNonOverlappingPixelHalfWidth```
       - The ``n_isotopes`` parameter of the ``App`` class can be used to select how many isotope m/z's should be used when calculating the mass ranges based on the SRL
       - When ``fill_non_denoised`` is TRUE the mass ranges outside the mass ranges determined by the mass range calculators are copied over to the de-noised file
 - De-noising parameters are managed by an instance of ``DenoiserBase`` class of  ``crane.denoiser``
 - The MS1 window can be de-noised with a different set of parameters than that of MS2 by passing a different instance of ``DenoiserBase`` class as the ``ms1_denoiser`` when initializing ``App`` class
 - ``window_name`` and ``include_ms1`` parameters of the ``App`` class can be used to select which windows are to be denoised

Please refer the crane_denoise_multicenter_*.py scripts in the case study section for an example.

## Running tests

```
pip install pytest
pytest crane
```

## References

1. Akila J Seneviratne, Sean Peters, David Clarke, Michael Dausmann, Michael Hecker, Brett Tully, Peter G Hains and Qing Zhong, "Improved identification and quantification of peptides in mass spectrometry data via chemical and random additive noise elimination (CRANE)"

2. Brett Tully, "Toffee – a highly efficient, lossless file format for DIA-MS". *Scientific Reports* 2020;10(1):8939