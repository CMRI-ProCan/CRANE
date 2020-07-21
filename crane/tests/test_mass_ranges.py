import os
import sys
import unittest
import pandas as pd
import numpy as np

from ..mass_ranges import (
    MassRangeCalculatorFullMap,
    MassRangeCalculatorFullMapSplit,
    MassRangeCalculatorNonOverlappingPPMFullWidth,
)
import toffee


class TestMassRangeCalculators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.path.realpath(__file__)
        dirname = os.path.dirname(path)
        cls.tof_fname = dirname +'/test_data/napedro_l120224_002_sw_ms2-001.tof'
        assert os.path.isfile(cls.tof_fname)

        swath_run = toffee.SwathRun(cls.tof_fname)
        cls.swath_map = swath_run.loadSwathMap('ms2-001')

        cls.mz_transformer = cls.swath_map.getMzTransformer()

    def test_MassRangeCalculatorFullMap(self):
        mz_transformer = self.mz_transformer

        # get expected ranges
        expected_lower = mz_transformer.lowerMzInIMSCoords()
        expected_upper = mz_transformer.upperMzInIMSCoords()

        # actual mass list isn't important
        mz_list = list()

        calculator = MassRangeCalculatorFullMap()
        ranges = calculator.calculate_ranges(self.swath_map, mz_list)
        self.assertEqual(1, len(ranges))
        mz_range = ranges[0]
        self.assertTrue(isinstance(mz_range, toffee.IMassOverChargeRange))
        self.assertEqual(expected_lower, mz_range.lowerMzInIMSCoords(mz_transformer))
        self.assertEqual(expected_upper, mz_range.upperMzInIMSCoords(mz_transformer))

        upper_offset = 2
        calculator = MassRangeCalculatorFullMap(upper_offset=upper_offset)
        ranges = calculator.calculate_ranges(self.swath_map, mz_list)
        self.assertEqual(1, len(ranges))
        mz_range = ranges[0]
        self.assertTrue(isinstance(mz_range, toffee.IMassOverChargeRange))
        self.assertEqual(expected_lower, mz_range.lowerMzInIMSCoords(mz_transformer))
        self.assertEqual(expected_upper - upper_offset, mz_range.upperMzInIMSCoords(mz_transformer))

    def test_MassRangeCalculatorFullMapSplit(self):
        mz_transformer = self.mz_transformer

        # get expected ranges
        expected_lower = self.mz_transformer.lowerMzInIMSCoords()
        expected_upper = self.mz_transformer.upperMzInIMSCoords()
        expected_total_delta = expected_upper - expected_lower
        # add 1 here because ranges are inclusive -- i.e. a delta of '0' would
        # result in a single pixel being extracted from the swath map
        expected_total_delta += 1

        split_size = 128
        calculator = MassRangeCalculatorFullMapSplit(split_size=split_size)

        # actual mass list isn't important
        mz_list = list()

        ranges = calculator.calculate_ranges(self.swath_map, mz_list)

        expected_num_ranges = expected_total_delta // split_size + 1
        self.assertEqual(expected_num_ranges, len(ranges))

        total_delta = 0
        for i, mz_range in enumerate(ranges):
            self.assertTrue(isinstance(mz_range, toffee.IMassOverChargeRange))
            delta = mz_range.upperMzInIMSCoords(mz_transformer) - mz_range.lowerMzInIMSCoords(mz_transformer)
            # add 1 here because ranges are inclusive -- i.e. a delta of '0' would
            # result in a single pixel being extracted from the swath map
            delta += 1

            # the last range might be smaller
            if i != len(ranges) - 1:
                self.assertEqual(split_size, delta)


    def test_MassRangeCalculatorNonOverlappingPPMFullWidth_1mz(self):
        '''
        test MassRangeCalculatorNonOverlappingPPMFullWidth on a hand full of arbitary mz values
        '''
        mz_transformer = self.mz_transformer

        ppm_full_width = 100
        split_size = 128
        calculator = MassRangeCalculatorNonOverlappingPPMFullWidth(
            ppm_full_width=ppm_full_width,
            split_size=split_size,
        )

        mz_list = [
            100,
        ]

        ranges = calculator.calculate_ranges(self.swath_map, mz_list)

        self.assertTrue(len(ranges) == 1)

        mz_range = ranges[0]
        self.assertTrue(isinstance(mz_range, toffee.IMassOverChargeRange))
        upper = mz_range.upperMzInIMSCoords(mz_transformer)
        lower = mz_range.lowerMzInIMSCoords(mz_transformer)
        self.assertTrue(upper - lower + 1 == split_size)
        expected_ind = self.mz_transformer.toIMSCoords(mz_list[0])
        self.assertTrue(expected_ind >= lower and expected_ind <= upper)

    def test_MassRangeCalculatorNonOverlappingPPMFullWidth_2mz_nonoverlapping(self):
        '''
        test MassRangeCalculatorNonOverlappingPPMFullWidth on a hand full of arbitary mz values
        '''
        mz_transformer = self.mz_transformer

        ppm_full_width = 100
        split_size = 128
        calculator = MassRangeCalculatorNonOverlappingPPMFullWidth(
            ppm_full_width=ppm_full_width,
            split_size=split_size,
        )

        mz_list = [
            100, 1000
        ]

        ranges = calculator.calculate_ranges(self.swath_map, mz_list)

        self.assertTrue(len(ranges) == 2)

        for i, mz_range in enumerate(ranges):
            self.assertTrue(isinstance(mz_range, toffee.IMassOverChargeRange))
            upper = mz_range.upperMzInIMSCoords(mz_transformer)
            lower = mz_range.lowerMzInIMSCoords(mz_transformer)
            self.assertTrue(upper - lower + 1 == split_size)
            expected_ind = self.mz_transformer.toIMSCoords(mz_list[i])
            self.assertTrue(expected_ind >= lower and expected_ind <= upper)

    def test_MassRangeCalculatorNonOverlappingPPMFullWidth_2mz_overlapping(self):
        '''
        test MassRangeCalculatorNonOverlappingPPMFullWidth on a hand full of arbitary mz values
        '''
        mz_transformer = self.mz_transformer

        ppm_full_width = 100
        split_size = 128
        calculator = MassRangeCalculatorNonOverlappingPPMFullWidth(
            ppm_full_width=ppm_full_width,
            split_size=split_size,
        )

        mz_list = [
            100, 100.001
        ]

        ranges = calculator.calculate_ranges(self.swath_map, mz_list)

        self.assertTrue(len(ranges) == 1)

        mz_range = ranges[0]
        self.assertTrue(isinstance(mz_range, toffee.IMassOverChargeRange))
        upper = mz_range.upperMzInIMSCoords(mz_transformer)
        lower = mz_range.lowerMzInIMSCoords(mz_transformer)
        self.assertTrue(upper - lower + 1 == split_size)

        for mz in mz_list:
            expected_ind = self.mz_transformer.toIMSCoords(mz)
            self.assertTrue(expected_ind >= lower and expected_ind <= upper)

    def test_MassRangeCalculatorNonOverlappingPPMFullWidth_nmz(self):
        '''
        test MassRangeCalculatorNonOverlappingPPMFullWidth on a hand full of arbitary mz values
        '''
        mz_transformer = self.mz_transformer

        ppm_full_width = 100
        split_size = 128
        calculator = MassRangeCalculatorNonOverlappingPPMFullWidth(
            ppm_full_width=ppm_full_width,
            split_size=split_size,
        )

        mz_list = [
            100, 100.001, 400, 400.002, 1000, 1000.05, 1000.1,
            1000.15, 1000.2, 1000.25, 1000.3, 1000.35, 1000.4,
            1000.45, 1000.5, 1000.55, 1000.6, 1000.65, 1000.7,
        ]

        # Calculate expected mz_ranges
        expected_ranges = pd.DataFrame(columns=['index_high', 'index_low', 'srl_inx'])

        # mz ranges shouldn't overlap
        for mz in sorted(mz_list):
            mz_high_limit = mz * (1 + ppm_full_width / 2000000)
            mz_low_limit = mz * (1 - ppm_full_width / 2000000)
            index_high_limit = self.mz_transformer.toIMSCoords(mz_high_limit)
            index_low_limit = self.mz_transformer.toIMSCoords(mz_low_limit)

            expected_ranges = expected_ranges.append(
                {
                    'index_high': index_high_limit,
                    'index_low': index_low_limit,
                    'srl_inx': self.mz_transformer.toIMSCoords(mz)
                },
                ignore_index=True,
            )

        ranges = calculator.calculate_ranges(self.swath_map, mz_list)

        for i, mz_range in enumerate(ranges):
            self.assertTrue(isinstance(mz_range, toffee.IMassOverChargeRange))
            upper = mz_range.upperMzInIMSCoords(mz_transformer)
            lower = mz_range.lowerMzInIMSCoords(mz_transformer)
            self.assertTrue(upper - lower < split_size * 2)
            self.assertTrue(upper - lower > split_size / 2)

        for index, row in expected_ranges.iterrows():
            mz_found = False
            for i, mz_range in enumerate(ranges):
                upper = mz_range.upperMzInIMSCoords(mz_transformer)
                lower = mz_range.lowerMzInIMSCoords(mz_transformer)
                if row[2] >= lower and row[2] <= upper:
                    mz_found = True
                    if row[1] < lower:
                        upper2 = ranges[i-1].upperMzInIMSCoords(mz_transformer)
                        lower2 = ranges[i-1].lowerMzInIMSCoords(mz_transformer)
                        self.assertTrue(lower >= lower2 and lower <= upper2)
                    if upper < row[0]:
                        upper2 = ranges[i+1].upperMzInIMSCoords(mz_transformer)
                        lower2 = ranges[i+1].lowerMzInIMSCoords(mz_transformer)
                        self.assertTrue(upper >= lower2 and upper <= upper2)
            self.assertTrue(mz_found)

    def test_MassRangeCalculatorNonOverlappingPPMFullWidth_srl(self):
        '''
        test MassRangeCalculatorNonOverlappingPPMFullWidth on mz values extracted from SGS SRL
        '''
        path = os.path.realpath(__file__)
        dirname = os.path.dirname(path)
        mz_transformer = self.mz_transformer

        ppm_full_width = 100
        split_size = 128
        calculator = MassRangeCalculatorNonOverlappingPPMFullWidth(
            ppm_full_width=ppm_full_width,
            split_size=split_size,
        )

        # SRL
        srl_paths = [
            dirname +'/test_data/OpenSWATH_SM4_GoldStandardAssayLibrary.tsv',
            dirname +'/test_data/OpenSWATH_SM4_iRT_AssayLibrary.tsv',
        ]

        library = list()
        for fname in srl_paths:
            assert os.path.isfile(fname)
            library.append(pd.read_table(fname))
        if len(library) == 1:
            library = library[0]
        else:
            library = pd.concat(library)

        # map the precursor m/z to the window in the the toffee file.
        lower_offset = 0.0
        upper_offset = 1.0
        swath_run = toffee.SwathRun(self.tof_fname)
        precursor_ms2_name_map = swath_run.mapPrecursorsToMS2Names(
            library.PrecursorMz.unique(),
            lower_offset,
            upper_offset
        )
        library['ms2Name'] = library.PrecursorMz.map(precursor_ms2_name_map)
        library = library[~library.ms2Name.isna()]
        library.set_index(['ms2Name', 'PeptideSequence'], inplace=True, drop=False)
        library.sort_index(inplace=True)

        mz_list = library.loc[pd.IndexSlice['ms2-001', :]].ProductMz.unique().tolist()

        # Calculate expected mz_ranges
        expected_ranges = pd.DataFrame(columns=['index_high', 'index_low', 'srl_inx'])

        # mz ranges shouldn't overlap
        for mz in sorted(mz_list):
            mz_high_limit = mz * (1 + ppm_full_width / 2000000)
            mz_low_limit = mz * (1 - ppm_full_width / 2000000)
            index_high_limit = self.mz_transformer.toIMSCoords(mz_high_limit)
            index_low_limit = self.mz_transformer.toIMSCoords(mz_low_limit)

            expected_ranges = expected_ranges.append(
                {
                    'index_high': index_high_limit,
                    'index_low': index_low_limit,
                    'srl_inx': self.mz_transformer.toIMSCoords(mz)
                },
                ignore_index=True,
            )

        ranges = calculator.calculate_ranges(self.swath_map, mz_list)

        for i, mz_range in enumerate(ranges):
            self.assertTrue(isinstance(mz_range, toffee.IMassOverChargeRange))
            upper = mz_range.upperMzInIMSCoords(mz_transformer)
            lower = mz_range.lowerMzInIMSCoords(mz_transformer)
            self.assertTrue(upper - lower < split_size * 2)
            self.assertTrue(upper - lower > split_size / 2)

        for index, row in expected_ranges.iterrows():
            mz_found = False
            for i, mz_range in enumerate(ranges):
                upper = mz_range.upperMzInIMSCoords(mz_transformer)
                lower = mz_range.lowerMzInIMSCoords(mz_transformer)
                if row[2] >= lower and row[2] <= upper:
                    mz_found = True
                    if row[1] < lower:
                        upper2 = ranges[i-1].upperMzInIMSCoords(mz_transformer)
                        lower2 = ranges[i-1].lowerMzInIMSCoords(mz_transformer)
                        self.assertTrue(lower >= lower2 and lower <= upper2)
                    if upper < row[0]:
                        upper2 = ranges[i+1].upperMzInIMSCoords(mz_transformer)
                        lower2 = ranges[i+1].lowerMzInIMSCoords(mz_transformer)
                        self.assertTrue(upper >= lower2 and upper <= upper2)
            self.assertTrue(mz_found)

    def test_complement_mass_range_calcualtor_nmz(self):
        '''
        test complemet_mass_ranges calculator using a hand full of arbitary mz values
        '''
        mz_transformer = self.mz_transformer
        ppm_full_width = 100
        split_size = 128

        calculator = MassRangeCalculatorNonOverlappingPPMFullWidth(
            ppm_full_width=ppm_full_width,
            split_size=split_size,
        )

        mz_list = [
            100, 100.001, 400, 400.002, 1000, 1000.05, 1000.1,
            1000.15, 1000.2, 1000.25, 1000.3, 1000.35, 1000.4,
            1000.45, 1000.5, 1000.55, 1000.6, 1000.65, 1000.7,
        ]

        ranges = calculator.calculate_ranges(self.swath_map, mz_list)
        complement_ranges = calculator.complement_mass_ranges(self.swath_map, ranges)

        full_mz = self.swath_map.massOverCharge()
        base_mass_over_charge_offset = mz_transformer.toIMSCoords(full_mz[0])

        # calculate a mask of combining the mass ranges and its complement
        mask = np.zeros(full_mz.shape)
        for mz_range in ranges:
            self.assertTrue(isinstance(mz_range, toffee.IMassOverChargeRange))
            lower = mz_range.lowerMzInIMSCoords(mz_transformer) - base_mass_over_charge_offset
            upper = mz_range.upperMzInIMSCoords(mz_transformer) - base_mass_over_charge_offset + 1
            mask[lower:upper] = 1

        for mz_range in complement_ranges:
            self.assertTrue(isinstance(mz_range, toffee.IMassOverChargeRange))
            lower = mz_range.lowerMzInIMSCoords(mz_transformer) - base_mass_over_charge_offset
            upper = mz_range.upperMzInIMSCoords(mz_transformer) - base_mass_over_charge_offset + 1
            self.assertTrue(np.sum(mask[lower:upper]) == 0)
            mask[lower:upper] = 1
        self.assertTrue(np.sum(mask) == mask.size)

    def test_complement_mass_range_calcualtor_srl(self):
        '''
        test complemet_mass_ranges calculator using a hand full of arbitary mz values
        '''
        path = os.path.realpath(__file__)
        dirname = os.path.dirname(path)
        mz_transformer = self.mz_transformer
        ppm_full_width = 100
        split_size = 128

        calculator = MassRangeCalculatorNonOverlappingPPMFullWidth(
            ppm_full_width=ppm_full_width,
            split_size=split_size,
        )

        # SRL
        srl_paths = [
            dirname +'/test_data/OpenSWATH_SM4_GoldStandardAssayLibrary.tsv',
            dirname +'/test_data/OpenSWATH_SM4_iRT_AssayLibrary.tsv',
        ]
        library = list()
        for fname in srl_paths:
            assert os.path.isfile(fname)
            library.append(pd.read_table(fname))
        if len(library) == 1:
            library = library[0]
        else:
            library = pd.concat(library)

        # map the precursor m/z to the window in the the toffee file.
        lower_offset = 0.0
        upper_offset = 1.0
        swath_run = toffee.SwathRun(self.tof_fname)
        precursor_ms2_name_map = swath_run.mapPrecursorsToMS2Names(
            library.PrecursorMz.unique(),
            lower_offset,
            upper_offset
        )
        library['ms2Name'] = library.PrecursorMz.map(precursor_ms2_name_map)
        library = library[~library.ms2Name.isna()]
        library.set_index(['ms2Name', 'PeptideSequence'], inplace=True, drop=False)
        library.sort_index(inplace=True)
        mz_list = library.loc[pd.IndexSlice['ms2-001', :]].ProductMz.unique().tolist()

        ranges = calculator.calculate_ranges(self.swath_map, mz_list)
        complement_ranges = calculator.complement_mass_ranges(self.swath_map, ranges)

        full_mz = self.swath_map.massOverCharge()
        base_mass_over_charge_offset = mz_transformer.toIMSCoords(full_mz[0])

        # calculate a mask of combining the mass ranges and its complement
        mask = np.zeros(full_mz.shape)
        for mz_range in ranges:
            self.assertTrue(isinstance(mz_range, toffee.IMassOverChargeRange))
            lower = mz_range.lowerMzInIMSCoords(mz_transformer) - base_mass_over_charge_offset
            upper = mz_range.upperMzInIMSCoords(mz_transformer) - base_mass_over_charge_offset + 1
            mask[lower:upper] = 1

        for mz_range in complement_ranges:
            self.assertTrue(isinstance(mz_range, toffee.IMassOverChargeRange))
            lower = mz_range.lowerMzInIMSCoords(mz_transformer) - base_mass_over_charge_offset
            upper = mz_range.upperMzInIMSCoords(mz_transformer) - base_mass_over_charge_offset + 1
            self.assertTrue(np.sum(mask[lower:upper]) == 0)
            mask[lower:upper] = 1
        self.assertTrue(np.sum(mask) == mask.size)
