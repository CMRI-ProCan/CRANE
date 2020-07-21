import os

import numpy as np
import pandas as pd
import scipy
import scipy.sparse
import datetime

import toffee

from tqdm import tqdm

from .srl import SpectralLibrary
from .denoiser import DenoiserBase
from .mass_ranges import MassRangeCalculatorBase


class App():
    """
    An application to run the denoising across a toffee file. This class will handle
    the sub-sampling as appropriate such that the mass ranges in the SRLs have data
    as defined by the mass range calculator
    """

    def __init__(self, tof_fname, srl_paths, denoiser, mass_range_calculator, ms1_denoiser=None, **kwargs):
        """
        :param str tof_fname: path to the raw toffee file, will raise exception if this file does
            not exist
        :param list[str] srl_paths: a list of paths to the SRL *.tsv files of interest. You must
            specify at least one path
        :param DenoiserBase denoiser: An implementation of :class:`DenoiserBase` that performs the
            noise removal based on 2D XIC intensities
        :param MassRangeCalculatorBase mass_range_calculator: An implementation of :class:`MassRangeCalculatorBase`
            that takes a list of m/z values and returns a list of the relevant :class:`toffee::IMassOverChargeRange`
            that can be passed through to extract data from toffee
        """
        # Name of the de-noised file
        denoised_fname = kwargs.get('denoised_fname', None)
        # Names of the MS2 windows to be denoised
        window_name = kwargs.get('window_name', None)
        # MS1 window will be de-noised if include_ms1 is True
        include_ms1 = kwargs.get('include_ms1', True)
        # Number of isotopes to be considered in the analysis
        n_isotopes = kwargs.get('n_isotopes', 4)
        # Raw data of the non denoised mass ranges are copied over if fill_non_denoised is True
        fill_non_denoised = kwargs.get('fill_non_denoised', True)

        # plugin classes
        self.denoiser = denoiser
        assert isinstance(self.denoiser, DenoiserBase)

        if ms1_denoiser is None:
            if include_ms1:
                # self.logger.debug('MS1 and MS2 will be denoised using the same algorithm')
                print('MS1 and MS2 will be denoised using the same algorithm')
            ms1_denoiser = denoiser
        self.ms1_denoiser = ms1_denoiser
        assert isinstance(self.ms1_denoiser, DenoiserBase)

        self.mass_range_calculator = mass_range_calculator
        assert isinstance(self.mass_range_calculator, MassRangeCalculatorBase)

        # TOF files
        self.tof_fname = tof_fname
        assert os.path.isfile(self.tof_fname)
        if denoised_fname is None:
            denoised_fname = self.tof_fname.replace('.tof', '.crane_v1.tof')
        self.denoised_fname = denoised_fname

        self.swath_run = toffee.SwathRun(self.tof_fname)

        # SRL
        self.srl = self._load_library(srl_paths, self.swath_run)

        # set up the windows
        if window_name is None:
            self.windows = sorted(self.srl.data.ms2Name.unique().tolist())
        else:
            self.windows = [window_name]
        if include_ms1:
            self.windows.append(toffee.ToffeeWriter.MS1_NAME)

        # Option to add isotopes to mass range calculations
        self.n_isotopes = n_isotopes
        assert isinstance(self.n_isotopes, int)

        self.fill_non_denoised = fill_non_denoised
        assert isinstance(self.fill_non_denoised, bool)

    #@profile
    def run(self):
        """
        Run the sub-sampling of the file
        """

        # set up the toffee writer
        tof_writer = toffee.ToffeeWriter(self.denoised_fname, self.swath_run.imsType())
        tof_writer.addHeaderMetadata(self.swath_run.header())

        # for each window, denoise and add to new toffee file
        for window_name in self.windows:
            print('Working on: ', window_name, ' of ', self.tof_fname)
            print(datetime.datetime.now())
            swath_map = self.swath_run.loadSwathMap(
                window_name,
            )

            # work out what m/z values need to be used
            # Calculate isotope mz values
            precursor_rows, product_rows = self.srl.calculate_isotope_mz(
                n_isotopes=self.n_isotopes,
                drop_other_cols=False
            )

            if swath_map.isMS1():
                mz_list = precursor_rows.IsotopeMz.unique().tolist()
            else:
                mz_list = product_rows.loc[product_rows.ms2Name == window_name].IsotopeMz.unique().tolist()

            # denoise and convert to a point cloud
            pcl = self._create_denoised_swath_map_as_point_cloud(swath_map, mz_list)

            # save to toffee file
            tof_writer.addPointCloud(pcl)
        print(datetime.datetime.now())

    @classmethod
    def _load_library(cls, srl_paths, swath_run, add_isotopes=True):
        """
        Load in the SRLs and match MZ values to their respective MS2 windows in the
        toffee file
        """
        assert len(srl_paths) > 0

        library_df = list()
        for fname in srl_paths:
            assert os.path.isfile(fname)
            library_df.append(pd.read_table(fname))
        if len(library_df) == 1:
            library_df = library_df[0]
        else:
            library_df = pd.concat(library_df)

        library = SpectralLibrary.init_from_df(library_df)

        # map the precursor m/z to the window in the the toffee file.
        lower_offset = 0.0
        upper_offset = 1.0
        library.add_ms2_windows(
            swath_run=swath_run,
            lower_offset=lower_offset,
            upper_offset=upper_offset,
        )

        return library

    def _create_denoised_swath_map_as_point_cloud(self, swath_map, mz_list):
        (base_mass_over_charge_offset,
         dok_sparse_matrix,
         mz_transformer) = self._create_dok_matrix(swath_map)

        # get a list of toffee.IMassOverChargeRange objects that can
        # be used to extract the XICs
        mz_ranges = self.mass_range_calculator.calculate_ranges(swath_map, mz_list)

        for mz_range in mz_ranges:
            xic = swath_map.extractedIonChromatogram(mz_range)
            if swath_map.isMS1():
                denoised_intensities = self.ms1_denoiser.apply(xic.intensities)
            else:
                denoised_intensities = self.denoiser.apply(xic.intensities)
            denoised_intensities = np.round(denoised_intensities).astype(np.uint32)
            sparse_intensities = scipy.sparse.dok_matrix(denoised_intensities)

            # the mass over charge values will have an offset relative to the start
            # of the global mass over charge
            offset = (
                mz_transformer.toIMSCoords(xic.massOverCharge[0]) -
                base_mass_over_charge_offset
            )

            # collect the sparse data from this chromatogram into the global one
            for k, v in sparse_intensities.items():
                assert v > 0
                dok_sparse_matrix[k[0], k[1] + offset] = v

        # Fill non de-noised mass ranges with raw data if fill_non_denoised is True
        if self.fill_non_denoised:
            # Calculate non de-noised mass ranges
            complement_mass_ranges = self.mass_range_calculator.complement_mass_ranges(swath_map, mz_ranges)

            for mz_range in complement_mass_ranges:
                xic_sparse = swath_map.extractedIonChromatogramSparse(mz_range)
                xic_intensities = xic_sparse.intensities.todok()

                # the mass over charge values will have an offset relative to the start
                # of the global mass over charge
                offset = (
                    mz_transformer.toIMSCoords(xic_sparse.massOverCharge[0]) -
                    base_mass_over_charge_offset
                )

                # collect the sparse data from this chromatogram into the global one
                for k, v in xic_intensities.items():
                    assert v > 0
                    dok_sparse_matrix[k[0], k[1] + offset] = v

        return self._create_point_cloud(
            self.swath_run,
            swath_map,
            dok_sparse_matrix,
        )

    def _create_dok_matrix(self, swath_map):
        """
        Generate a Dictionary of Keys type sparse matrix that spans the range of the swath map,
        in addition to the lower square root (m/z) index
        """
        mz_transformer = swath_map.getMzTransformer()

        # this offset is to balance the need for toffee to have an offset that
        # matches the lower m/z value, with the need for our sparse matrices
        # to start their index at 0
        base_mass_over_charge_offset = mz_transformer.lowerMzInIMSCoords()

        # create a holding sparse matrix for all of the data to be added to
        # use a dictionary of keys style sparse matrix for fast updating
        # of values -- it can be converted to another format once this
        # stage has been completed
        sparse_matrix = scipy.sparse.dok_matrix(
            (swath_map.retentionTime().size, swath_map.massOverCharge().size),
            dtype=np.uint32
        )

        return base_mass_over_charge_offset, sparse_matrix, mz_transformer

    def _create_point_cloud(self, swath_run, swath_map, dok_matrix):
        """
        Convert the data into a form that can be saved to toffee
        """
        swath_map_spectrum_access = swath_run.loadSwathMapSpectrumAccess(swath_map.name())
        assert isinstance(swath_run, toffee.SwathRun)
        assert isinstance(swath_map, toffee.SwathMap)
        assert isinstance(swath_map_spectrum_access, toffee.SwathMapSpectrumAccessBase)
        assert isinstance(dok_matrix, scipy.sparse.dok_matrix)
        
        min_allowable_version = toffee.Version.combineFileFormatVersions(1, 1)
        file_version = toffee.Version.combineFileFormatVersions(
            swath_run.formatMajorVersion(),
            swath_run.formatMinorVersion(),
        )
        if file_version < min_allowable_version:
            raise ValueError(
                f'File format is invalid: {file_version} < {min_allowable_version}',
            )

        pcl = toffee.ScanDataPointCloud()
        pcl.name = swath_map.name()
        pcl.windowLower = swath_map.precursorLowerMz()
        pcl.windowCenter = swath_map.precursorCenterMz()
        pcl.windowUpper = swath_map.precursorUpperMz()
        pcl.scanCycleTime = swath_map.scanCycleTime()
        pcl.firstScanRetentionTimeOffset = swath_map.firstScanRetentionTimeOffset()

        sparse_matrix = dok_matrix.tocsr()
        sparse_matrix.sort_indices()
        pcl.setIntensityFromNumpy(sparse_matrix.data)
        pcl.imsProperties.sliceIndex = sparse_matrix.indptr[1:]

        rt = swath_map.retentionTime()
        num_scans = rt.shape[0]

        mz_transformer = swath_map.getMzTransformer()
        pcl.imsProperties.medianAlpha = mz_transformer.imsAlpha()
        pcl.imsProperties.medianBeta = mz_transformer.imsBeta()
        pcl.imsProperties.gamma = mz_transformer.imsGamma()

        # add the lower IMS coord offset back in
        sparse_matrix.indices += mz_transformer.lowerMzInIMSCoords()

        alpha = [0.0] * num_scans
        beta = [0.0] * num_scans
        for i in range(num_scans):
            scan_mz_transformer = swath_map_spectrum_access.getMzTransformer(i)
            alpha[i] = scan_mz_transformer.imsAlpha()
            beta[i] = scan_mz_transformer.imsBeta()

        pcl.imsProperties.alpha = alpha
        pcl.imsProperties.beta = beta

        pcl.imsProperties.setCoordinateFromNumpy(sparse_matrix.indices)

        return pcl