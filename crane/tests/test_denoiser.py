import os
import sys
import numpy as np
import unittest
import pywt
import math

from ..denoiser import (
    DenoiserNoOp,
    Crane,
    AdaptiveCrane,
    Thresholding,
    Transform,
    Adaptive_Level_Selection,
)


class TestDenoiser(unittest.TestCase):

    def test_DenoiserNoOp(self):
        intensities = np.random.uniform(size=(2, 2))
        denoiser = DenoiserNoOp()
        denoised_intensities = denoiser.apply(intensities)
        np.testing.assert_array_equal(intensities, denoised_intensities)
    
    def test_Crane_1(self):
        """
        When denoising options are all turned off the output of the denoiser should be the same as the input
        """
        path = os.path.realpath(__file__)
        dirname = os.path.dirname(path)
        intensities = np.load(dirname +'/test_data/test_raw_SGS_001_MS1_442.821.npy')
        denoiser = Crane(
            transform_technique=Transform.NoTransform,
            wavelet_method='db2',
            levels=7,
            apply_hrmc=False,
            split_data=False,
            thresholding_technique=Thresholding.NoThresholding,
            smooth_coeff_suppression=0,
            artifact_filter_with_thresholded_XIC=True,
        )
        denoised_intensities = denoiser.apply(intensities)
        expected_denoised_intensities = intensities

        # denoised intensities should be of the same dimension as the raw
        self.assertEqual(intensities.shape[0], denoised_intensities.shape[0])
        self.assertEqual(intensities.shape[1], denoised_intensities.shape[1])

        # denoised intensities should be non negative
        self.assertTrue(denoised_intensities.min() >= 0)

        np.testing.assert_array_equal(expected_denoised_intensities, denoised_intensities)

    def test_Crane_2(self):
        """
        When denoising options are all turned off the output of the denoiser should be the same as the input
        test_Crane_2 is different from test_Crane_1 in that split_data=True
        """
        path = os.path.realpath(__file__)
        dirname = os.path.dirname(path)
        intensities = np.load(dirname +'/test_data/test_raw_SGS_001_MS1_442.821.npy')
        denoiser = Crane(
            transform_technique=Transform.NoTransform,
            wavelet_method='db2',
            levels=7,
            apply_hrmc=False,
            split_data=True,
            thresholding_technique=Thresholding.NoThresholding,
            smooth_coeff_suppression=0,
            artifact_filter_with_thresholded_XIC=True,
        )
        denoised_intensities = denoiser.apply(intensities)
        expected_denoised_intensities = intensities

        # denoised intensities should be of the same dimension as the raw
        self.assertEqual(intensities.shape[0], denoised_intensities.shape[0])
        self.assertEqual(intensities.shape[1], denoised_intensities.shape[1])

        # denoised intensities should be non negative
        self.assertTrue(denoised_intensities.min() >= 0)

        np.testing.assert_array_equal(expected_denoised_intensities, denoised_intensities)

    def test_Crane_3(self):
        """
        When denoising options are all turned off the output of the denoiser should be the same as the input
        test_Crane_3 is different from test_Crane_2 in that apply_log=True
        """
        path = os.path.realpath(__file__)
        dirname = os.path.dirname(path)
        intensities = np.load(dirname +'/test_data/test_raw_SGS_001_MS1_442.821.npy')
        denoiser = Crane(
            transform_technique=Transform.Log,
            wavelet_method='db2',
            levels=7,
            apply_hrmc=False,
            split_data=True,
            thresholding_technique=Thresholding.NoThresholding,
            smooth_coeff_suppression=0,
            artifact_filter_with_thresholded_XIC=True,
        )
        denoised_intensities = denoiser.apply(intensities)
        expected_denoised_intensities = intensities

        # denoised intensities should be of the same dimension as the raw
        self.assertEqual(intensities.shape[0], denoised_intensities.shape[0])
        self.assertEqual(intensities.shape[1], denoised_intensities.shape[1])

        # denoised intensities should be non negative
        self.assertTrue(denoised_intensities.min() >= 0)

        np.testing.assert_array_equal(expected_denoised_intensities, denoised_intensities)
        
    def test_Crane_4(self):
        """
        When denoising options are all turned off the output of the denoiser should be the same as the input
        test_Crane_4 is different from test_Crane_3 in that artifact_filter_with_thresholded_XIC=False
        """
        path = os.path.realpath(__file__)
        dirname = os.path.dirname(path)
        intensities = np.load(dirname +'/test_data/test_raw_SGS_001_MS1_442.821.npy')
        denoiser = Crane(
            transform_technique=Transform.Log,
            wavelet_method='db2',
            levels=7,
            apply_hrmc=False,
            split_data=True,
            thresholding_technique=Thresholding.NoThresholding,
            smooth_coeff_suppression=0,
            artifact_filter_with_thresholded_XIC=False,
        )
        denoised_intensities = denoiser.apply(intensities)
        expected_denoised_intensities = intensities

        # denoised intensities should be of the same dimension as the raw
        self.assertEqual(intensities.shape[0], denoised_intensities.shape[0])
        self.assertEqual(intensities.shape[1], denoised_intensities.shape[1])

        # denoised intensities should be non negative
        self.assertTrue(denoised_intensities.min() >= 0)

        np.testing.assert_array_equal(expected_denoised_intensities, denoised_intensities)
    
    def test_AdaptiveCrane_1(self):
        """
        When denoising options are all turned off the output of the denoiser should be the same as the input
        """
        path = os.path.realpath(__file__)
        dirname = os.path.dirname(path)
        intensities = np.load(dirname +'/test_data/test_raw_SGS_001_MS1_442.821.npy')
        denoiser = AdaptiveCrane(
            transform_technique=Transform.NoTransform,
            wavelet_method='db2',
            max_level=7,
            min_level=4,
            adaptive_level_selection_technique=Adaptive_Level_Selection.Diagonal_Power_Mean_3,
            apply_hrmc=False,
            split_data=False,
            thresholding_technique=Thresholding.NoThresholding,
            smooth_coeff_suppression=0,
            artifact_filter_with_thresholded_XIC=True,
            max_denoise=False,
        )
        denoised_intensities = denoiser.apply(intensities)
        expected_denoised_intensities = intensities

        # denoised intensities should be of the same dimension as the raw
        self.assertEqual(intensities.shape[0], denoised_intensities.shape[0])
        self.assertEqual(intensities.shape[1], denoised_intensities.shape[1])

        # denoised intensities should be non negative
        self.assertTrue(denoised_intensities.min() >= 0)

        np.testing.assert_array_equal(expected_denoised_intensities, denoised_intensities)

    def test_AdaptiveCrane_2(self):
        """
        When denoising options are all turned off the output of the denoiser should be the same as the input
        test_AdaptiveCrane_2 is different from test_AdaptiveCrane_1 in that split_data=True
        """
        path = os.path.realpath(__file__)
        dirname = os.path.dirname(path)
        intensities = np.load(dirname +'/test_data/test_raw_SGS_001_MS1_442.821.npy')
        denoiser = AdaptiveCrane(
            transform_technique=Transform.NoTransform,
            wavelet_method='db2',
            max_level=7,
            min_level=4,
            adaptive_level_selection_technique=Adaptive_Level_Selection.Diagonal_Power_Mean_3,
            apply_hrmc=False,
            split_data=True,
            thresholding_technique=Thresholding.NoThresholding,
            smooth_coeff_suppression=0,
            artifact_filter_with_thresholded_XIC=True,
            max_denoise=False,
        )
        denoised_intensities = denoiser.apply(intensities)
        expected_denoised_intensities = intensities

        # denoised intensities should be of the same dimension as the raw
        self.assertEqual(intensities.shape[0], denoised_intensities.shape[0])
        self.assertEqual(intensities.shape[1], denoised_intensities.shape[1])

        # denoised intensities should be non negative
        self.assertTrue(denoised_intensities.min() >= 0)

        np.testing.assert_array_equal(expected_denoised_intensities, denoised_intensities)

    def test_AdaptiveCrane_3(self):
        """
        When denoising options are all turned off the output of the denoiser should be the same as the input
        test_AdaptiveCrane_3 is different from test_AdaptiveCrane_2 in that apply_log=True
        """
        path = os.path.realpath(__file__)
        dirname = os.path.dirname(path)
        intensities = np.load(dirname +'/test_data/test_raw_SGS_001_MS1_442.821.npy')
        denoiser = AdaptiveCrane(
            transform_technique=Transform.Log,
            wavelet_method='db2',
            max_level=7,
            min_level=4,
            adaptive_level_selection_technique=Adaptive_Level_Selection.Diagonal_Power_Mean_3,
            apply_hrmc=False,
            split_data=True,
            thresholding_technique=Thresholding.NoThresholding,
            smooth_coeff_suppression=0,
            artifact_filter_with_thresholded_XIC=True,
            max_denoise=False,
        )
        denoised_intensities = denoiser.apply(intensities)
        expected_denoised_intensities = intensities

        # denoised intensities should be of the same dimension as the raw
        self.assertEqual(intensities.shape[0], denoised_intensities.shape[0])
        self.assertEqual(intensities.shape[1], denoised_intensities.shape[1])

        # denoised intensities should be non negative
        self.assertTrue(denoised_intensities.min() >= 0)

        np.testing.assert_array_equal(expected_denoised_intensities, denoised_intensities)
        
    def test_AdaptiveCrane_4(self):
        """
        When denoising options are all turned off the output of the denoiser should be the same as the input
        test_AdaptiveCrane_4 is different from test_AdaptiveCrane_3 in that artifact_filter_with_thresholded_XIC=False
        """
        path = os.path.realpath(__file__)
        dirname = os.path.dirname(path)
        intensities = np.load(dirname +'/test_data/test_raw_SGS_001_MS1_442.821.npy')
        denoiser = AdaptiveCrane(
            transform_technique=Transform.Log,
            wavelet_method='db2',
            max_level=7,
            min_level=4,
            adaptive_level_selection_technique=Adaptive_Level_Selection.Diagonal_Power_Mean_3,
            apply_hrmc=False,
            split_data=True,
            thresholding_technique=Thresholding.NoThresholding,
            smooth_coeff_suppression=0,
            artifact_filter_with_thresholded_XIC=False,
            max_denoise=False,
        )
        denoised_intensities = denoiser.apply(intensities)
        expected_denoised_intensities = intensities

        # denoised intensities should be of the same dimension as the raw
        self.assertEqual(intensities.shape[0], denoised_intensities.shape[0])
        self.assertEqual(intensities.shape[1], denoised_intensities.shape[1])

        # denoised intensities should be non negative
        self.assertTrue(denoised_intensities.min() >= 0)

        np.testing.assert_array_equal(expected_denoised_intensities, denoised_intensities)
        
    def test_AdaptiveCrane_5(self):
        """
        When apply_hrmc=False, the output of Crane should equal that of AdaptiveCrane
        """
        path = os.path.realpath(__file__)
        dirname = os.path.dirname(path)
        intensities = np.load(dirname +'/test_data/test_raw_SGS_001_MS1_442.821.npy')
        denoiser_crane = Crane(
            transform_technique=Transform.NoTransform,
            wavelet_method='db2',
            levels=7,
            apply_hrmc=True,
            split_data=True,
            thresholding_technique=Thresholding.VisuShrink,
            smooth_coeff_suppression=-1,
            artifact_filter_with_thresholded_XIC=True,
        )
        expected_denoised_intensities = denoiser_crane.apply(intensities)
        denoiser_adaptive_crane = AdaptiveCrane(
            transform_technique=Transform.NoTransform,
            wavelet_method='db2',
            max_level=7,
            min_level=7,
            adaptive_level_selection_technique=Adaptive_Level_Selection.Diagonal_Power_Mean_3,
            apply_hrmc=True,
            split_data=True,
            thresholding_technique=Thresholding.VisuShrink,
            smooth_coeff_suppression=-1,
            artifact_filter_with_thresholded_XIC=True,
            max_denoise=False,
        )
        denoised_intensities = denoiser_adaptive_crane.apply(intensities)
        
        # denoised intensities should be of the same dimension as the raw
        self.assertEqual(expected_denoised_intensities.shape[0], denoised_intensities.shape[0])
        self.assertEqual(expected_denoised_intensities.shape[1], denoised_intensities.shape[1])

        # denoised intensities should be non negative
        self.assertTrue(denoised_intensities.min() >= 0)

        np.testing.assert_array_equal(expected_denoised_intensities, denoised_intensities)

    def test_split_n_udwt(self):
        """
        Check whether the split_n_udwt functions return the lists of expected lengths
        """
        levels = 7
        
        path = os.path.realpath(__file__)
        dirname = os.path.dirname(path)
        intensities = np.load(dirname +'/test_data/test_raw_SGS_001_MS1_442.821.npy')
        denoiser = Crane(
            transform_technique=Transform.Log,
            wavelet_method='db2',
            levels=levels,
            apply_hrmc=False,
            split_data=True,
            thresholding_technique=Thresholding.NoThresholding,
            smooth_coeff_suppression=0,
            artifact_filter_with_thresholded_XIC=False,
            max_denoise=False,
        )
        dwt_coeffs_splits, pad_lengths, split_dim = denoiser.split_n_udwt(
            intensities,
            levels=levels,
            wavelet=pywt.Wavelet('db2'),
        )

        expected_num_splits = int(np.floor(intensities.T.shape[1] / split_dim)) + 1
        self.assertTrue(expected_num_splits == len(dwt_coeffs_splits))

        for inx in range(expected_num_splits):
            dwt_coeffs_split = dwt_coeffs_splits[inx]
            wavelet_levels = len(dwt_coeffs_split)
            self.assertTrue(levels == wavelet_levels)

            for inx2 in range(wavelet_levels):
                dwt_coeffs_split_level = dwt_coeffs_split[inx2]
                self.assertTrue(len(dwt_coeffs_split_level) == 2)

                cA = dwt_coeffs_split_level[0]
                self.assertTrue(cA.shape[0] == split_dim and cA.shape[1] == split_dim)
                (cH, cV, cD) = dwt_coeffs_split_level[1]
                self.assertTrue(cH.shape[0] == split_dim and cH.shape[1] == split_dim)
                self.assertTrue(cV.shape[0] == split_dim and cV.shape[1] == split_dim)
                self.assertTrue(cD.shape[0] == split_dim and cD.shape[1] == split_dim)

    def test_udwt(self):
        """
        Check whether the udwt functions return the lists of expected lengths
        """
        levels = 7

        path = os.path.realpath(__file__)
        dirname = os.path.dirname(path)
        intensities = np.load(dirname +'/test_data/test_raw_SGS_001_MS1_442.821.npy')
        denoiser = Crane(
            transform_technique=Transform.Log,
            wavelet_method='db2',
            levels=levels,
            apply_hrmc=False,
            split_data=False,
            thresholding_technique=Thresholding.NoThresholding,
            smooth_coeff_suppression=0,
            artifact_filter_with_thresholded_XIC=False,
            max_denoise=False,
        )
        dwt_coeffs, pad_len_col, pad_len_row = denoiser.udwt(
            intensities,
            levels=levels,
            wavelet=pywt.Wavelet('db2'),
        )

        wavelet_levels = len(dwt_coeffs)
        self.assertTrue(levels == wavelet_levels)

        expected_width = intensities.shape[1] + pad_len_row
        expected_length = intensities.shape[0] + pad_len_col
        for inx in range(wavelet_levels):
            dwt_coeffs_level = dwt_coeffs[inx]
            self.assertTrue(len(dwt_coeffs_level) == 2)

            cA = dwt_coeffs_level[0]
            self.assertTrue(cA.shape[0] == expected_width and cA.shape[1] == expected_length)
            (cH, cV, cD) = dwt_coeffs_level[1]
            self.assertTrue(cH.shape[0] == expected_width and cH.shape[1] == expected_length)
            self.assertTrue(cV.shape[0] == expected_width and cV.shape[1] == expected_length)
            self.assertTrue(cD.shape[0] == expected_width and cD.shape[1] == expected_length)
            
    def test_SUREShrink(self):
        """
        Test if the SUREShrink threshold calculation function returns the correct threshold
        """
        def cal_SUREShrink_threshold_slow(coeff, noise_sigma, visu_threshold):
            """
            Calculates the hybrid VisuShrink plus SUREShrink threshold based on the subband coefficients and noise variance
            """
            if np.count_nonzero(coeff) == 0:
                # Empty coefficient matrix
                return 0

            coeff_sort = np.sort(np.absolute(coeff.flatten()))
            non_zero_length = np.count_nonzero(coeff_sort)
            coeff_length = len(coeff_sort)
            noise_sigma_square  = noise_sigma**2

            s_d = np.sum(np.square(coeff_sort)-1)/coeff_length
            gamma_d = math.sqrt((math.log(coeff_length,2)**3)/coeff_length)

            if s_d > gamma_d:
                if non_zero_length < coeff_length:
                    coeff_sort_short = coeff_sort[coeff_length-non_zero_length-1:coeff_length]
                else:
                    coeff_sort_short = coeff_sort

                SURE = np.empty(coeff_sort_short.shape)
                coeff_square = np.square(coeff_sort_short)
                coeff_short_length = len(coeff_sort_short)

                for i in range(0,coeff_short_length,1):
                    threshold = coeff_sort_short[i]
                    threshold_square = threshold ** 2
                    small_coeff_index = coeff_sort_short <= threshold
                    num_small_coeff = np.sum(small_coeff_index)
                    
                    SURE[i] = coeff_short_length * noise_sigma_square 
                    - 2 * noise_sigma_square * num_small_coeff 
                    + threshold_square * (coeff_short_length-num_small_coeff)
                    + np.sum(coeff_square[small_coeff_index])

                threshold = coeff_sort_short[np.argmin(SURE)]

                if threshold > visu_threshold:
                    return visu_threshold
                else:
                    return threshold
            else:
                return visu_threshold

        levels = 7
        tolerance = 0.001

        path = os.path.realpath(__file__)
        dirname = os.path.dirname(path)
        intensities = np.load(dirname +'/test_data/test_raw_SGS_001_MS1_442.821.npy')
        denoiser = Crane(
            transform_technique=Transform.Log,
            wavelet_method='db2',
            levels=levels,
            apply_hrmc=False,
            split_data=False,
            thresholding_technique=Thresholding.NoThresholding,
            smooth_coeff_suppression=0,
            artifact_filter_with_thresholded_XIC=False,
            max_denoise=False,
        )
        dwt_coeffs, pad_len_col, pad_len_row = denoiser.udwt(
            intensities,
            levels=levels,
            wavelet=pywt.Wavelet('db2'),
        )

        (cH, cV, cD) = dwt_coeffs[-1][1]
        img_size = cD.shape[0] * cD.shape[1]
        visu_threshold, noise_sigma = denoiser.cal_VisuShrink_threshold(cD, img_size)
        
#         for j in range(1, levels + 1, 1):
        for j in range(1, 1 + 1, 1):
            (cH, cV, cD) = dwt_coeffs[-j][1]
            cH_SURE_threshold_slow = cal_SUREShrink_threshold_slow(cH, noise_sigma, visu_threshold)
            cH_SURE_threshold = denoiser.cal_SUREShrink_threshold(cH, noise_sigma, visu_threshold)
            if cH_SURE_threshold_slow > 0:
                self.assertTrue(np.absolute((cH_SURE_threshold_slow-cH_SURE_threshold)/cH_SURE_threshold_slow) < tolerance)
            
            cV_SURE_threshold_slow = cal_SUREShrink_threshold_slow(cV, noise_sigma, visu_threshold)
            cV_SURE_threshold = denoiser.cal_SUREShrink_threshold(cV, noise_sigma, visu_threshold)
            if cV_SURE_threshold_slow > 0:
                self.assertTrue(np.absolute((cV_SURE_threshold_slow-cV_SURE_threshold)/cV_SURE_threshold_slow) < tolerance)
            
            cD_SURE_threshold_slow = cal_SUREShrink_threshold_slow(cD, noise_sigma, visu_threshold)
            cD_SURE_threshold = denoiser.cal_SUREShrink_threshold(cD, noise_sigma, visu_threshold)
            if cD_SURE_threshold_slow > 0:
                self.assertTrue(np.absolute((cD_SURE_threshold_slow-cD_SURE_threshold)/cD_SURE_threshold_slow) < tolerance)

