import numpy as np
import scipy.interpolate
import pywt
import logging
import enum
import math
import pybeads

class Transform(enum.Enum):
    NoTransform = 1
    Log = 2
    Anscombe1 = 3 # Forward transform y = 2 * sqrt(x + 3/8), inverse x = (y/2)^2 - 3/8
    Anscombe2 = 4 # Forward transform y = 2 * sqrt(x + 3/8), inverse x = (y/2)^2 - 1/8


class Thresholding(enum.Enum):
    VisuShrink = 1
    BayesShrink = 2
    NoThresholding = 3
    SUREShrink = 4
    ModifiedVisuShrink = 5


class Adaptive_Level_Selection(enum.Enum):
    No_Adaptive_selection = 1
    Diagonal_Power_Mean_3 = 2
    Diagonal_Power_Mean_4 = 3
    Approximate_Power_Mean_3 = 4
    Diagonal_Sigma_3 = 5
    K_param = 6
    Jarque_Bera = 7
    K_n_sparsity = 8
    JB_n_sparsity = 9
    

class DenoiserBase():
    """
    An abstract base class to control the interface for implementations of
    Denoisers
    """

    def apply(self, intensities):
        raise NotImplementedError('Called from base class')


class DenoiserNoOp(DenoiserBase):
    """
    A no-operation class that simply returns the intensities without applying
    any algorithms
    """

    def apply(self, intensities):
        return intensities


class CraneBase(DenoiserBase):
    """
    An abstract base class to hold common fuctions used by versions of Crane
    """

    @classmethod
    def cal_VisuShrink_threshold(cls, coeff, img_size):
        """
        Calculate the VisuShrink threshold given the level one wavelet
        coefficients (coeff) and the length of the signal
        """

        if np.count_nonzero(coeff) == 0:
            # Empty coefficient matrix
            return 0, 0

        # numpy flattens matrices before performing functions such as calculating the element wise
        # absolute so the coefficient matrix is flattened to form a vector first to avoid indexing errors
        coeff_flat = coeff.flatten()
        coeff_flat_abs = np.absolute(coeff_flat)

        # MAD of the coefficients of the level 1 decomposition
        absolute_deviation_from_median = np.absolute(coeff_flat - np.median(coeff_flat))
        noiseMAD = np.median(absolute_deviation_from_median)

        # Estimate noise SD
        noiseSigma = noiseMAD / 0.67449

        # calculate threshold
        threshold = noiseSigma * np.sqrt(2 * np.log(img_size))

        return threshold, noiseSigma
    
    @classmethod
    def cal_modified_VisuShrink_threshold(cls, visu_threshold, img_size):
        """
        Calculate the modified VisuShrink threshold 
        """
        return visu_threshold / np.sqrt(img_size)

    @classmethod
    def cal_BayesShrink_threshold(cls, coeff, noise_sigma):
        """
        Calculates the subband BayesShrink threshold based on the subband coefficients and noise variance
        """
        if np.count_nonzero(coeff) == 0:
            # Empty coefficient matrix
            return 0

        noise_sigma_square = noise_sigma ** 2
        coeff_sort = np.sort(np.absolute(coeff.flatten()))
        coeff_variance = np.sum(np.square(coeff_sort)) / len(coeff_sort)
        if coeff_variance > noise_sigma_square:
            sigma_x = np.sqrt(coeff_variance - noise_sigma_square)
            return noise_sigma_square / sigma_x
        else:
            return coeff_sort.max()

    @classmethod
    def cal_SUREShrink_threshold(cls, coeff, noise_sigma, visu_threshold):
        """
        Calculates the hybrid VisuShrink plus SUREShrink threshold based on the subband coefficients and noise variance
        David L. Donoho and lain M. Johnstone, "Adapting to Unknown Smoothness via Wavelet Shrinkage", 
        Journal of the American Statistical Association, Vol. 90, No. 432 (Dec., 1995), pp. 1200-1224
        """
        if np.count_nonzero(coeff) == 0 or np.abs(visu_threshold) < 1e-20:
            # Empty coefficient matrix
            return visu_threshold
        
        def SURE(noise_sigma_square, coeff_length, coeff, coeff_square, index_array):
            SURE_array = np.empty(index_array.shape)
            for i in range(len(index_array)):
                threshold = coeff[index_array[i]]
                threshold_square = threshold ** 2
                small_coeff_index = coeff <= threshold
                num_small_coeff = np.sum(small_coeff_index)

                SURE_array[i] = coeff_length * noise_sigma_square 
                - 2 * noise_sigma_square * num_small_coeff 
                + threshold_square * (coeff_length-num_small_coeff)
                + np.sum(coeff_square[small_coeff_index])

            return SURE_array
        def cal_index_2 (index_lo, index_hi):
            return index_lo + math.floor((index_hi - index_lo) / 1.61803398875)
        
        def cal_index_1 (index_lo, index_hi):
            return index_hi - math.floor((index_hi - index_lo) / 1.61803398875)

        noise_sigma_square = noise_sigma ** 2
        coeff_sort = np.sort(np.absolute(coeff.flatten()))
        non_zero_length = np.count_nonzero(coeff_sort)
        coeff_length = len(coeff_sort)
        coeff_sort_square = np.square(coeff_sort)
        
        s_d = np.sum(coeff_sort_square-1)/coeff_length
        gamma_d = math.sqrt((math.log(coeff_length,2)**3)/coeff_length)

        if s_d > gamma_d:
            if non_zero_length < coeff_length:
                coeff_sort_short = coeff_sort[coeff_length-non_zero_length-1:coeff_length]
                coeff_sort_square_short = coeff_sort_square[coeff_length-non_zero_length-1:coeff_length]
            else:
                coeff_sort_short = coeff_sort
                coeff_sort_square_short = coeff_sort_square
            coeff_short_length = len(coeff_sort_short)
            
            # Use golden section search to find the SURE minimizer (unimodel function)
            converged = 0
            index_lo = 0
            index_hi = coeff_short_length - 1
            index_1 = cal_index_1(index_lo, index_hi)
            index_2 = cal_index_2(index_lo, index_hi)
            index_array = np.array([index_lo, index_1, index_2, index_hi])
            SURE_array = SURE(noise_sigma_square, coeff_short_length, coeff_sort_short, coeff_sort_square_short, index_array)
            tolarence = SURE_array[0]/1000

            while converged == 0:
                min_index = index_array[np.argmin(SURE_array)]
                if index_array[1] - index_array[2] <= 1:
                    converged = 1
                if np.abs(SURE_array[1] - SURE_array[2]) < tolarence:
                    converged = 1
                    min_index = math.floor((index_array[1] + index_array[2])/2)
                threshold = coeff_sort_short[min_index]
                if converged == 0:
                    if SURE_array[1] > SURE_array[2]:
                        index_lo = index_1
                        index_1 = cal_index_1(index_lo, index_hi)
                        index_2 = cal_index_2(index_lo, index_hi)
                        SURE_array[0:3] = SURE(
                            noise_sigma_square,
                            coeff_short_length,
                            coeff_sort_short,
                            coeff_sort_square_short,
                            [index_lo, index_1, index_2])
                    else:
                        index_hi = index_2
                        index_1 = cal_index_1(index_lo, index_hi)
                        index_2 = cal_index_2(index_lo, index_hi)
                        SURE_array[1:4] = SURE(
                            noise_sigma_square,
                            coeff_short_length,
                            coeff_sort_short,
                            coeff_sort_square_short,
                            [index_1, index_2, index_hi])
                    index_array = np.array([index_lo, index_1, index_2, index_hi])
                    if np.argmin(index_array) > 0:
                        converged = 1
                    if np.argmax(index_array) != 3:
                        converged = 1
                    if index_array[1] > index_array[2]:
                        converged = 1
            if threshold > visu_threshold:
                return visu_threshold
            else:
                return threshold
        else:
            return visu_threshold

    @classmethod
    def cal_adaptive_level_selection_criterion(cls, dwt_coeffs, adaptive_level_selection_technique):
        dwt_n_level_coeffs = dwt_coeffs[0]
        n = dwt_n_level_coeffs[0].shape[0] * dwt_n_level_coeffs[0].shape[1]
        if adaptive_level_selection_technique == Adaptive_Level_Selection.Diagonal_Power_Mean_3:
            (cH, cV, cD) = dwt_n_level_coeffs[1]
            return (((cD - cD.mean()) ** 3).sum()) / n
        if adaptive_level_selection_technique == Adaptive_Level_Selection.Diagonal_Power_Mean_4:
            (cH, cV, cD) = dwt_n_level_coeffs[1]
            return (((cD - cD.mean()) ** 4).sum()) / n
        if adaptive_level_selection_technique == Adaptive_Level_Selection.Approximate_Power_Mean_3:
            cA = dwt_n_level_coeffs[0]
            return (((cA - cA.mean()) ** 3).sum()) / n
        if adaptive_level_selection_technique == Adaptive_Level_Selection.Diagonal_Sigma_3:
            (cH, cV, cD) = dwt_n_level_coeffs[1]
            var = (((cD - cD.mean()) ** 2).sum()) / n
            return math.sqrt(var ** 3)
        if adaptive_level_selection_technique == Adaptive_Level_Selection.K_param:
            K_a = []
            for i in range(1, len(dwt_coeffs) + 1, 1):
                cA = dwt_coeffs[-i][0]
                mu_4 = (((cA - cA.mean()) ** 4).sum()) / n
                var = (((cA - cA.mean()) ** 2).sum()) / n
                sigma_4 = var ** 2
                K_a.append(mu_4 / sigma_4)
            return (K_a[-2]-K_a[-1])/K_a[0]
        if adaptive_level_selection_technique == Adaptive_Level_Selection.Jarque_Bera:
            JB_a = []
            for i in range(1, len(dwt_coeffs) + 1, 1):
                cA = dwt_coeffs[-i][0]
                mu_4 = (((cA - cA.mean()) ** 4).sum()) / n
                var = (((cA - cA.mean()) ** 2).sum()) / n
                sigma_4 = var ** 2
                K = mu_4 / sigma_4
                mu_3 = (((cA - cA.mean()) ** 3).sum()) / n
                sigma_3 = math.sqrt(var ** 3)
                S = mu_3 / sigma_3
                JB_a.append(n / 6 * ((S ** 2) + (1 / 4 * ((K - 3) ** 2))))
            return (JB_a[-2]-JB_a[-1])/JB_a[0]
        if adaptive_level_selection_technique == Adaptive_Level_Selection.K_n_sparsity:
            K_a = []
            for i in range(1, len(dwt_coeffs) + 1, 1):
                cA = dwt_coeffs[-i][0]
                mu_4 = (((cA - cA.mean()) ** 4).sum()) / n
                var = (((cA - cA.mean()) ** 2).sum()) / n
                sigma_4 = var ** 2
                K_a.append(mu_4 / sigma_4)

            cA = dwt_n_level_coeffs[0]
            threshold = 1
            cA[cA<threshold] == 0
            sparsity = 1 - (np.count_nonzero(cA)/n)
            
            return [(K_a[-2]-K_a[-1])/K_a[0], sparsity]
        if adaptive_level_selection_technique == Adaptive_Level_Selection.JB_n_sparsity:
            JB_a = []
            for i in range(1, len(dwt_coeffs) + 1, 1):
                cA = dwt_coeffs[-i][0]
                mu_4 = (((cA - cA.mean()) ** 4).sum()) / n
                var = (((cA - cA.mean()) ** 2).sum()) / n
                sigma_4 = var ** 2
                K = mu_4 / sigma_4
                mu_3 = (((cA - cA.mean()) ** 3).sum()) / n
                sigma_3 = math.sqrt(var ** 3)
                S = mu_3 / sigma_3
                JB_a.append(n / 6 * ((S ** 2) + (1 / 4 * ((K - 3) ** 2))))
            
            cA = dwt_n_level_coeffs[0]
            threshold = 1
            cA[cA<threshold] == 0
            sparsity = 1 - (np.count_nonzero(cA)/n)
            
            return [(JB_a[-2]-JB_a[-1])/JB_a[0], sparsity]

    
    @classmethod
    def udwt(cls, intensities, levels, wavelet):
        """
        Undecimated wavelet transform (UDWT)
        The UDWT implementation of pywavelets requier the signal to be a multiple of 2^levels
        so the extracted chromatogram is padded before undecimated wavelet transformtion via pywavelets
        """
        # calculate the maximum number of levels of wavelet decomposition
        min_data_length = min(intensities.T.shape[0], intensities.T.shape[1])
        max_level = round(np.log2(min_data_length))
        max_level = max_level + 1

        if levels > max_level:
            levels = max_level

        # The number of columns and rows should be a multiple of 2**levels
        required_length = 2 ** levels
        pad_len_col = required_length - (intensities.T.shape[1] % required_length)
        pad_len_row = required_length - (intensities.T.shape[0] % required_length)
        padded_xic = np.pad(intensities.T, ((pad_len_row, 0), (pad_len_col, 0)), 'edge')
        dwt_coeffs = pywt.swt2(padded_xic, wavelet, level=levels, start_level=0)

        return dwt_coeffs, pad_len_col, pad_len_row
    
    @classmethod
    def adaptive_udwt(cls, intensities, min_level, max_level, wavelet, adaptive_level_selection_technique):
        # calculate the maximum number of levels of wavelet decomposition
        min_data_length = min(intensities.T.shape[0], intensities.T.shape[1])
        maximum_levels = round(np.log2(min_data_length))
        maximum_levels = maximum_levels + 1

        if max_level > maximum_levels:
            max_level = maximum_levels
            
        # The number of columns and rows should be a multiple of 2**levels
        required_length = 2 ** max_level
        pad_len_col = required_length - (intensities.T.shape[1] % required_length)
        pad_len_row = required_length - (intensities.T.shape[0] % required_length)
        padded_xic = np.pad(intensities.T, ((pad_len_row, 0), (pad_len_col, 0)), 'edge')
        dwt_coeffs = pywt.swt2(padded_xic, wavelet, level=min_level, start_level=0)
        adaptive_level_selection_criterion = cls.cal_adaptive_level_selection_criterion(
            dwt_coeffs,
            adaptive_level_selection_technique,
        )
        try_another_level = True
        optimal_wavelet_level = min_level
        for inx in range(min_level+1, max_level+1, 1):
#             next_dwt_coeffs = pywt.swt2(dwt_coeffs[0][0], wavelet, level=1, start_level=0)
            next_dwt_coeffs = pywt.swt2(padded_xic, wavelet, level=inx, start_level=0)
            next_adaptive_level_selection_criterion = cls.cal_adaptive_level_selection_criterion(
                next_dwt_coeffs,
                adaptive_level_selection_technique,
            )
            if adaptive_level_selection_technique == Adaptive_Level_Selection.K_param:
                try_another_level = (adaptive_level_selection_criterion > 0.09)
                
            elif adaptive_level_selection_technique == Adaptive_Level_Selection.Jarque_Bera:
                try_another_level = (adaptive_level_selection_criterion > 0.09)
                
            elif adaptive_level_selection_technique == Adaptive_Level_Selection.K_n_sparsity:
                try_another_level = (adaptive_level_selection_criterion[0] > 0.09 and adaptive_level_selection_criterion[1] > 0.001)

            elif adaptive_level_selection_technique == Adaptive_Level_Selection.JB_n_sparsity:
                try_another_level = (adaptive_level_selection_criterion[0] > 0.09 and adaptive_level_selection_criterion[1] > 0.001)

            else:
                try_another_level = (next_adaptive_level_selection_criterion >= adaptive_level_selection_criterion)

            if try_another_level:
                optimal_wavelet_level = inx
    #                 dwt_coeffs = next_dwt_coeffs + dwt_coeffs
                dwt_coeffs = next_dwt_coeffs
            else:
                break
            adaptive_level_selection_criterion = next_adaptive_level_selection_criterion
        return dwt_coeffs, pad_len_col, pad_len_row, optimal_wavelet_level 
    
    @classmethod
    def inverse_udwt(cls, coeffs, wavelet, pad_len_col, pad_len_row):
        """
        Inverse Undecimated wavelet transform (UDWT)
        """
        # Inverse wavelet transformation
        filtered_coeff = pywt.iswt2(coeffs, wavelet)
        
        # Remove padding
        intensities = filtered_coeff[pad_len_row:, pad_len_col:]
        
        return intensities

    @classmethod
    def split_n_udwt(cls, intensities, levels, wavelet):
        """
        extracted chromatogram is split into sections and padded to form a square of dimension that is a multiple of
        2^levels and then eash split is undecimated wavelet transformed using pywavelets
        """
        # calculate the maximum number of levels of wavelet decomposition
        min_data_length = min(intensities.T.shape[0], intensities.T.shape[1])
        max_level = round(np.log2(min_data_length))
        max_level = max_level + 1

        if levels > max_level:
            levels = max_level
        required_length = 2 ** levels

        if intensities.T.shape[0] > intensities.T.shape[1]:
            # If number of rows is higher than the number od columns, pad the columns
            # to get a multiple of of the requiered length and split the rows
            pad_len_col = required_length - (intensities.T.shape[1] % required_length)
            split_dim = intensities.T.shape[1] + pad_len_col
            num_splits = int(np.floor(intensities.T.shape[0] / split_dim)) + 1
            pad_len_row = 0
            dwt_coeffs_splits = []
            for inx in range(num_splits-1):
                padded_xic = np.pad(
                    intensities.T[inx * split_dim: (inx + 1) * split_dim, :],
                    ((pad_len_row, 0), (pad_len_col, 0)),
                    'edge'
                )
                dwt_coeffs_splits.append(pywt.swt2(padded_xic, wavelet, level=levels, start_level=0))

            # if the number of rows was not a multiple of the split_dim length,
            # recalculate the padding for the remainder and introduce a new split
            if intensities.T.shape[0] > (num_splits - 1) * split_dim:
                pad_len_col2 = pad_len_col
                pad_len_row2 = split_dim - (intensities.T.shape[0] - split_dim * (num_splits - 1))
                padded_xic = np.pad(
                    intensities.T[(num_splits - 1) * split_dim:, :],
                    ((pad_len_row2, 0), (pad_len_col2, 0)),
                    'edge'
                )
                dwt_coeffs_splits.append(pywt.swt2(padded_xic, wavelet, level=levels, start_level=0))
            else:
                num_splits = num_splits - 1
                pad_len_col2 = pad_len_col
                pad_len_row2 = 0
        else:
            # If number of columns is higher than the number od rows, pad the rows
            # to get a multiple of of the requiered length and split the columns
            pad_len_row = required_length - (intensities.T.shape[0] % required_length)
            split_dim = intensities.T.shape[0] + pad_len_row
            num_splits = int(np.floor(intensities.T.shape[1] / split_dim)) + 1
            pad_len_col = 0
            dwt_coeffs_splits = []
            for inx in range(num_splits-1):
                padded_xic = np.pad(
                    intensities.T[:, inx * split_dim: (inx + 1) * split_dim],
                    ((pad_len_row, 0), (pad_len_col, 0)),
                    'edge'
                )
                dwt_coeffs_splits.append(pywt.swt2(padded_xic, wavelet, level=levels, start_level=0))

            # if the number of columns was not a multiple of the split_dim length,
            # recalculate the padding for the remainder and introduce a new split
            if intensities.T.shape[1] > (num_splits - 1) * split_dim:
                pad_len_row2 = pad_len_row
                pad_len_col2 = split_dim - (intensities.T.shape[1] - split_dim * (num_splits - 1))
                padded_xic = np.pad(
                    intensities.T[:, (num_splits - 1) * split_dim:],
                    ((pad_len_row2, 0), (pad_len_col2, 0)),
                    'edge'
                )
                dwt_coeffs_splits.append(pywt.swt2(padded_xic, wavelet, level=levels, start_level=0))
            else:
                num_splits = num_splits - 1
                pad_len_col2 = 0
                pad_len_row2 = pad_len_row

        pad_lengths = np.array([pad_len_row, pad_len_col, pad_len_row2, pad_len_col2])
        return dwt_coeffs_splits, pad_lengths, split_dim
    
    @classmethod
    def cal_split_optimal_wavelet_level(cls, dwt_coeffs_splits, min_level, max_level, adaptive_level_selection_technique):
        optimal_wavelet_level = []
        num_splits = len(dwt_coeffs_splits)
        for inx in range(num_splits):
            dwt_coeffs_split = dwt_coeffs_splits[inx]
            wavelet_levels = len(dwt_coeffs_split)
            if min_level < wavelet_levels:
                split_optimal_wavelet_level = min_level
                adaptive_level_selection_criterion = cls.cal_adaptive_level_selection_criterion(
                    [dwt_coeffs_split[-min_level]],
                    adaptive_level_selection_technique,
                )
                if wavelet_levels >= max_level:
                    iter_max = max_level
                else:
                    iter_max = wavelet_levels
                try_another_level = True
                for j in range(min_level+1, iter_max+1, 1):
                    next_adaptive_level_selection_criterion = cls.cal_adaptive_level_selection_criterion(
                        [dwt_coeffs_split[-j]],
                        adaptive_level_selection_technique,
                    )
                    if adaptive_level_selection_technique == Adaptive_Level_Selection.K_param:
                        try_another_level = (adaptive_level_selection_criterion > 0.09)

                    elif adaptive_level_selection_technique == Adaptive_Level_Selection.Jarque_Bera:
                        try_another_level = (adaptive_level_selection_criterion > 0.09)

                    elif adaptive_level_selection_technique == Adaptive_Level_Selection.K_n_sparsity:
                        try_another_level = (adaptive_level_selection_criterion[0] > 0.09 and adaptive_level_selection_criterion[1] > 0.001)

                    elif adaptive_level_selection_technique == Adaptive_Level_Selection.JB_n_sparsity:
                        try_another_level = (adaptive_level_selection_criterion[0] > 0.09 and adaptive_level_selection_criterion[1] > 0.001)

                    else:
                        try_another_level = (next_adaptive_level_selection_criterion >= adaptive_level_selection_criterion)
                    
                    if try_another_level:
                        split_optimal_wavelet_level = j
                    else:
                        break
                    adaptive_level_selection_criterion = next_adaptive_level_selection_criterion
            else:
                split_optimal_wavelet_level = wavelet_levels
            optimal_wavelet_level.append(split_optimal_wavelet_level)
        return optimal_wavelet_level

    @classmethod
    def adaptive_split_n_udwt(cls, intensities, min_level, max_level, wavelet, adaptive_level_selection_technique):
        dwt_coeffs_splits, pad_lengths, split_dim = cls.split_n_udwt(
            intensities,
            levels=max_level,
            wavelet=wavelet,
        )
        optimal_wavelet_level = cls.cal_split_optimal_wavelet_level(
            dwt_coeffs_splits,
            min_level=min_level,
            max_level=max_level,
            adaptive_level_selection_technique=adaptive_level_selection_technique,
        )
        return dwt_coeffs_splits, pad_lengths, split_dim, optimal_wavelet_level

    @classmethod
    def inverse_udwt_n_combine_splits(cls, coeffs, wavelet, pad_lengths, split_dim, raw_data_shape):
        """
        Inverse Undecimated wavelet transform (UDWT) and reconstruct XIC from the splits
        """
        num_splits = len(coeffs)
        pad_len_row = pad_lengths[0]
        pad_len_col = pad_lengths[1]
        pad_len_row2 = pad_lengths[2]
        pad_len_col2 = pad_lengths[3]
        intensities = np.empty(raw_data_shape)
        
        for inx in range(num_splits):
            # Inverse wavelet transformation
            filtered_coeff = pywt.iswt2(coeffs[inx], wavelet)
            
            # If data were split before wavelet transformation then reassemble
            if raw_data_shape[0] > raw_data_shape[1]:
                if inx == num_splits-1:
                    unpadded_filtered_coeff_split = filtered_coeff[pad_len_row2:, pad_len_col2:]
                    intensities[(num_splits-1)*split_dim:, :] = unpadded_filtered_coeff_split
                else:
                    unpadded_filtered_coeff_split = filtered_coeff[pad_len_row:, pad_len_col:]
                    intensities[inx*split_dim:(inx+1)*split_dim, :] = unpadded_filtered_coeff_split
            else:
                if inx == num_splits-1:
                    unpadded_filtered_coeff_split = filtered_coeff[pad_len_row2:, pad_len_col2:]
                    intensities[:, (num_splits-1)*split_dim:] = unpadded_filtered_coeff_split
                else:
                    unpadded_filtered_coeff_split = filtered_coeff[pad_len_row:, pad_len_col:]
                    intensities[:, inx*split_dim:(inx+1)*split_dim] = unpadded_filtered_coeff_split
        return intensities
        
    @classmethod
    def adaptive_inverse_udwt_n_combine_splits(
        cls,
        coeffs,
        wavelet,
        pad_lengths,
        split_dim,
        optimal_wavelet_level,
        raw_data_shape
    ):
        num_splits = len(coeffs)
        assert num_splits == len(optimal_wavelet_level), "There should be an optimal wavelet decomposition level per split"
        pad_len_row = pad_lengths[0]
        pad_len_col = pad_lengths[1]
        pad_len_row2 = pad_lengths[2]
        pad_len_col2 = pad_lengths[3]
        intensities = np.empty(raw_data_shape)
        for inx in range(num_splits):
            optimal_split_wavelet_level = optimal_wavelet_level[inx]
            # Inverse wavelet transformation
            split_coeffs = coeffs[inx]
            filtered_coeff = pywt.iswt2(split_coeffs[-optimal_split_wavelet_level:], wavelet)
            
            # If data were split before wavelet transformation then reassemble
            if raw_data_shape[0] > raw_data_shape[1]:
                if inx == num_splits-1:
                    unpadded_filtered_coeff_split = filtered_coeff[pad_len_row2:, pad_len_col2:]
                    intensities[(num_splits-1)*split_dim:, :] = unpadded_filtered_coeff_split
                else:
                    unpadded_filtered_coeff_split = filtered_coeff[pad_len_row:, pad_len_col:]
                    intensities[inx*split_dim:(inx+1)*split_dim, :] = unpadded_filtered_coeff_split
            else:
                if inx == num_splits-1:
                    unpadded_filtered_coeff_split = filtered_coeff[pad_len_row2:, pad_len_col2:]
                    intensities[:, (num_splits-1)*split_dim:] = unpadded_filtered_coeff_split
                else:
                    unpadded_filtered_coeff_split = filtered_coeff[pad_len_row:, pad_len_col:]
                    intensities[:, inx*split_dim:(inx+1)*split_dim] = unpadded_filtered_coeff_split
        return intensities
    
    @classmethod
    def hrmc_splits(cls, coeff_splits):
        """
        Given the UDWT of the splits of an extracted ion chromatogram calculate the level wise
        overall median and the row wise median of the horizontal coefficients and adjust the
        row wise median to equal the overall median to remove chemical noise
        This function assumes that before wavelet transformation the RT of the intensity matrix
        varied across columns and was that same across rows so that the chemical noise is
        captured in the horizontal component
        This function also assumes that the data were split along axis 1
        """
        num_splits = len(coeff_splits)
        num_levels = len(coeff_splits[0])
        num_rows = len(coeff_splits[0][0][0])
        for l in range(num_levels):
            (cH, cV, cD) = coeff_splits[0][l][1]
            concat_cH = cH
            for s in range(1, num_splits, 1):
                (cH, cV, cD) = coeff_splits[s][l][1]
                concat_cH = np.concatenate((concat_cH, cH), axis=1)
            level_median = np.median(concat_cH)
            row_median = np.median(concat_cH, axis=1)
            for r in range(num_rows):
                if row_median[r] != level_median:
                    concat_cH[r, :] = concat_cH[r, :] - (np.ones(concat_cH.shape[1]) * (row_median[r] - level_median))
            for s in range(num_splits):
                (cH, cV, cD) = coeff_splits[s][l][1]
                coeff_splits[s][l] = [coeff_splits[s][l][0], (concat_cH[:, s * num_rows: (s + 1) * num_rows], cV, cD)]
        return coeff_splits

    @classmethod
    def hrmc(cls, coeffs):
        """
        Given the UDWT of an extracted ion chromatogram calculate the level wise
        overall median and the row wise median of the horizontal coefficients and adjust the
        row wise median to equal the overall median to remove chemical noise
        This function assumes that before wavelet transformation the RT of the intensity matrix
        varied across columns and was that same across rows so that the chemical noise is
        captured in the horizontal component
        """
        num_levels = len(coeffs)
        num_rows = len(coeffs[0][0])
        for l in range(num_levels):
            (cH, cV, cD) = coeffs[l][1]
            level_median = np.median(cH)
            row_median = np.median(cH, axis=1)
            for r in range(num_rows):
                if row_median[r] != level_median:
                    cH[r, :] = cH[r, :] - (np.ones(cH.shape[1]) * (row_median[r] - level_median))
            coeffs[l] = [coeffs[l][0], (cH, cV, cD)]
        return coeffs

    @classmethod
    def calculate_threshold(cls, coeffs, thresholding_technique, visu_threshold, noise_sigma, img_size):
        if thresholding_technique == Thresholding.VisuShrink:
            return visu_threshold

        if thresholding_technique == Thresholding.BayesShrink:
            return cls.cal_BayesShrink_threshold(coeffs, noise_sigma)

        if thresholding_technique == Thresholding.SUREShrink:
            return cls.cal_SUREShrink_threshold(coeffs, noise_sigma, visu_threshold)
        
        if thresholding_technique == Thresholding.ModifiedVisuShrink:
            return cls.cal_modified_VisuShrink_threshold(visu_threshold, img_size)

        return 0

    @classmethod
    def apply_transform(cls, intensities, transform_technique):
        transform_intensities = intensities

        if transform_technique == Transform.Log:
            transform_intensities = np.log10(intensities + 1)
        if transform_technique in [Transform.Anscombe1, Transform.Anscombe2]:
            transform_intensities = 2* np.sqrt(intensities + (3/8))

        return transform_intensities

    @classmethod
    def apply_inverse_transform(cls, transform_intensities, transform_technique):
        intensities = transform_intensities

        if transform_technique == Transform.Log:
            intensities = np.round(10 ** transform_intensities - 1)
        if transform_technique == Transform.Anscombe1:
            intensities = np.round((transform_intensities / 2) ** 2 - (3 / 8))
        if transform_technique == Transform.Anscombe2:
            intensities = np.round((transform_intensities / 2) ** 2 - (1 / 8))

        return intensities

    def apply(self, intensities):
        raise NotImplementedError('Called from base class')


class Crane(CraneBase):
    """
    A denoising approach based on wavelets. In the default setting for each strip of data six levels of wavelet 
    coefficients are calculated via db2. Hard thresholding is performed on each of the horizontal, vertical and 
    diagonal coefficients with the threshold selected via VisuShrink. Raw median correction of the horizontal 
    coefficients remove chemical noise. The artifact filter uses the inverse transform of thresholded data if hrmc 
    produces artifacts 
    """

    def __init__(
        self,
        transform_technique=Transform.NoTransform,
        wavelet_method='db2',
        levels=6,
        apply_hrmc=True,
        split_data=False,
        thresholding_technique=Thresholding.VisuShrink,
        smooth_coeff_suppression=-1,
        artifact_filter_with_thresholded_XIC=True,
        max_denoise=False,
    ):
        """
        :param transform_technique Transform: instance of the class Transform specifying the
        transformation technique to be used before wavelet transform
        :param wavelet_method string: name of the wavelet used for transformation,
        this should be one of the wavelets implemented in pywavelets
        :param levels int: number of levels of wavelet decomposition
        :param apply_hrmc boolean: if true horizontal coefficient's raw median corrected to
        remove chemical noise
        :param split_data boolean: if true the data is subdevided into smaller squares before denoising
        :param thresholding_technique Thresholding: instance of the class Thresholding specifying the
        thresholding technique,
        :param smooth_coeff_suppression int: number of levels the smooth coefficients are supressed and set to zero
        e.g.
        if smooth_coeff_suppression = 2 then the first and second level smooth wavelet coefficients are set to zero
        if smooth_coeff_suppression = 0 then none of the smooth wavelet coefficients are set to zero
        if smooth_coeff_suppression = -1 then all smooth wavelet coefficients are set to zero
        :param artifact_filter_with_thresholded_XIC boolean: if True artifact filter uses the inverse transform 
        of thresholded data if hrmc produces artifacts
        :max_denoise boolean: If true use the inverse transform of the thresholded and the HRMC coefficients to 
        get maximum denoising
        """
        self.transform_technique = transform_technique
        assert isinstance(self.transform_technique, Transform)
        assert wavelet_method in pywt.wavelist()
        self.wavelet = pywt.Wavelet(wavelet_method)
        self.apply_hrmc = apply_hrmc  # hmrc = horizontal row median correction
        assert isinstance(self.apply_hrmc, bool)
        self.levels = levels
        assert isinstance(self.levels, int)
        assert self.levels > 0, 'Number of levels of wavelet decomposition should be greater than 0'
        self.split_data = split_data
        assert isinstance(self.split_data, bool)
        self.thresholding_technique = thresholding_technique
        assert isinstance(self.thresholding_technique, Thresholding)
        self.smooth_coeff_suppression = smooth_coeff_suppression
        assert isinstance(self.smooth_coeff_suppression, int)
        assert self.smooth_coeff_suppression >= -1
        self.artifact_filter_with_thresholded_XIC=artifact_filter_with_thresholded_XIC
        assert isinstance(self.artifact_filter_with_thresholded_XIC, bool)
        self.max_denoise = max_denoise
        assert isinstance(self.max_denoise, bool)

    def apply(self, intensities):
        """
        Given the extracted ion chromatogram intensities this function returns the denoised data via
        crane version 3
        """

        if np.count_nonzero(intensities) == 0:
            # Empty intensities matrix
            logging.debug('Empty intensities matrix')
            return intensities

        split_size = 2 ** self.levels
        # mass_ranges class should ensure that strips are not too narrow
        assert intensities.shape[1] >= split_size / 2

        intensities = self.apply_transform(intensities, self.transform_technique)

        if self.split_data:
            # Split XIC and perform UDWT
            dwt_coeffs_splits, pad_lengths, split_dim = self.split_n_udwt(
                intensities,
                levels=self.levels,
                wavelet=self.wavelet,
            )
            num_splits = len(dwt_coeffs_splits)
        else:
            # perform UDWT
            dwt_coeffs, pad_len_col, pad_len_row = self.udwt(
                intensities,
                levels=self.levels,
                wavelet=self.wavelet
            )
            dwt_coeffs_splits = [dwt_coeffs]
            num_splits = 1

        threshold_mode = 'hard'
        thresholded_coeffs = dwt_coeffs_splits.copy()

        for inx in range(num_splits):
            dwt_coeffs_split = dwt_coeffs_splits[inx]
            wavelet_levels = len(dwt_coeffs_split)

            # Estimate noise standard deviation
            (cH, cV, cD) = dwt_coeffs_split[-1][1]
            img_size = cD.shape[0] * cD.shape[1]
            visu_threshold, noise_sigma = self.cal_VisuShrink_threshold(cD, img_size)
            thresholded_coeff = dwt_coeffs_split.copy()

            # De-noise via selected thresholding technique
            for j in range(1, wavelet_levels + 1, 1):
                (cH, cV, cD) = dwt_coeffs_split[-j][1]
                threshold = self.calculate_threshold(cH, self.thresholding_technique, visu_threshold, noise_sigma, img_size)
                # When cHf is made zero for any level the performance degrades
                cHf = pywt.threshold(cH, threshold, mode=threshold_mode)
                threshold = self.calculate_threshold(cV, self.thresholding_technique, visu_threshold, noise_sigma, img_size)
                cVf = pywt.threshold(cV, threshold, mode=threshold_mode)
                threshold = self.calculate_threshold(cD, self.thresholding_technique, visu_threshold, noise_sigma, img_size)
                cDf = pywt.threshold(cD, threshold, mode=threshold_mode)
                if j <= self.smooth_coeff_suppression or self.smooth_coeff_suppression == -1:
                    thresholded_coeff[-j] = [np.zeros((cD.shape[0], cD.shape[1])), (cHf, cVf, cDf)]
                else:
                    thresholded_coeff[-j] = [dwt_coeffs_split[-j][0], (cHf, cVf, cDf)]
            thresholded_coeffs[inx] = thresholded_coeff
        
        thresholded_coeffs_copy = thresholded_coeffs.copy()
        
        if self.split_data:
            # Perform inverse UDWT and reconstruct data from splits
            xic_thresholded = self.inverse_udwt_n_combine_splits(
                coeffs=thresholded_coeffs_copy,
                wavelet=self.wavelet,
                pad_lengths=pad_lengths,
                split_dim=split_dim,
                raw_data_shape=intensities.T.shape,
            )
        else:
            # perform inverse UDWT
            xic_thresholded = self.inverse_udwt(
                coeffs=thresholded_coeffs_copy[0],
                wavelet=self.wavelet,
                pad_len_col=pad_len_col,
                pad_len_row=pad_len_row,
            )
        xic_thresholded[xic_thresholded < 0] = 0
        mask_threshold = xic_thresholded > intensities.T
        xic_thresholded[mask_threshold] = intensities.T[mask_threshold]
                
        # horizontal row median correction (hrmc)
        if self.apply_hrmc:
            if self.split_data:
                thresholded_hrmc_coeffs = self.hrmc_splits(thresholded_coeffs)
                # Perform inverse UDWT and reconstruct data from splits
                xic_hrmc_denoised = self.inverse_udwt_n_combine_splits(
                    coeffs=thresholded_hrmc_coeffs,
                    wavelet=self.wavelet,
                    pad_lengths=pad_lengths,
                    split_dim=split_dim,
                    raw_data_shape=intensities.T.shape,
                )
            else:
                thresholded_hrmc_coeffs = [self.hrmc(thresholded_coeffs[0])]
                # perform inverse UDWT
                xic_hrmc_denoised = self.inverse_udwt(
                    coeffs=thresholded_hrmc_coeffs[0],
                    wavelet=self.wavelet,
                    pad_len_col=pad_len_col,
                    pad_len_row=pad_len_row,
                )
        else:
            thresholded_hrmc_coeffs = thresholded_coeffs.copy()
            xic_hrmc_denoised = xic_thresholded

        # Remove artefacts - Denoised has to be less than or equal to the original and non negative
        xic_hrmc_denoised[xic_hrmc_denoised < 0] = 0
        mask = xic_hrmc_denoised > intensities.T
        if self.artifact_filter_with_thresholded_XIC and self.apply_hrmc:
            xic_hrmc_denoised[mask] = xic_thresholded[mask]
        else:
            xic_hrmc_denoised[mask] = intensities.T[mask]

        xic_final_denoised = xic_hrmc_denoised.copy()
        
        if self.max_denoise:
            mask_final = xic_final_denoised > xic_thresholded
            xic_final_denoised[mask_final] = xic_thresholded[mask_final]

        xic_final_denoised = self.apply_inverse_transform(xic_final_denoised, self.transform_technique)

        return xic_final_denoised.T


class AdaptiveCrane(CraneBase):
    """
    A denoising approach based on wavelets. In the default setting for each strip of data is decomposed adaptively
    between 4 and 8 levels of decomposition based on power mean of exponent 3 of the diagonal coefficients via db2. 
    Hard thresholding is performed on each of the horizontal, vertical and diagonal coefficients with the threshold 
    selected via VisuShrink. Raw median correction of the horizontal coefficients remove chemical noise. The 
    artifact filter uses the inverse transform of thresholded data if hrmc produces artifacts. All smooth 
    coefficients are suppressed
    """
    def __init__(
        self,
        transform_technique=Transform.NoTransform,
        wavelet_method='db2',
        max_level=8,
        min_level=4,
        adaptive_level_selection_technique=Adaptive_Level_Selection.Diagonal_Power_Mean_3,
        apply_hrmc=True,
        split_data=False,
        thresholding_technique=Thresholding.VisuShrink,
        smooth_coeff_suppression=-1,
        artifact_filter_with_thresholded_XIC=True,
        max_denoise=False,
    ):
        """
        :param transform_technique Transform: instance of the class Transform specifying the
        transformation technique to be used before wavelet transform
        :param wavelet_method string: name of the wavelet used for transformation,
        this should be one of the wavelets implemented in pywavelets
        :param max_level int: maximum number of levels of wavelet decomposition
        :param min_level int: minimum number of levels of wavelet decomposition
        :param adaptive_level_selection_technique Adaptive_Level_Selection: instance of the class
        Adaptive_Level_Selection that define the technique of wavelet level selection
        :param apply_hrmc boolean: if true horizontal coefficient's raw median corrected to
        remove chemical noise
        :param split_data boolean: if true the data is subdevided into smaller squares before denoising
        :param thresholding_technique Thresholding: instance of the class Thresholding specifying the
        thresholding technique,
        :param smooth_coeff_suppression int: number of levels the smooth coefficients are supressed and set to zero
        e.g.
        if smooth_coeff_suppression = 2 then the first and second level smooth wavelet coefficients are set to zero
        if smooth_coeff_suppression = 0 then none of the smooth wavelet coefficients are set to zero
        if smooth_coeff_suppression = -1 then all smooth wavelet coefficients are set to zero
        if smooth_coeff_suppression > optimat number of levels of decomposition then all smooth wavelet coefficients 
        are set to zero
        :param artifact_filter_with_thresholded_XIC boolean: if True artifact filter uses the inverse transform 
        of thresholded data if hrmc produces artifacts
        :max_denoise boolean: If true use the inverse transform of the thresholded and the HRMC coefficients to 
        get maximum denoising
        """
        self.transform_technique = transform_technique
        assert isinstance(self.transform_technique, Transform)
        assert wavelet_method in pywt.wavelist()
        self.wavelet = pywt.Wavelet(wavelet_method)
        self.max_level = max_level
        assert isinstance(self.max_level, int)
        assert self.max_level > 0, 'Maximum number of levels of wavelet decomposition should be greater than 0'
        self.min_level = min_level
        assert isinstance(self.min_level, int)
        assert self.min_level > 0, 'Minimum number of levels of wavelet decomposition should be greater than 0'
        assert self.max_level >= self.min_level, 'Maximum number of levels of wavelet decomposition should be greater than the minimum'
        self.adaptive_level_selection_technique = adaptive_level_selection_technique
        assert isinstance(self.adaptive_level_selection_technique, Adaptive_Level_Selection)
        self.apply_hrmc = apply_hrmc  # hmrc = horizontal row median correction
        assert isinstance(self.apply_hrmc, bool)
        self.split_data = split_data
        assert isinstance(self.split_data, bool)
        self.thresholding_technique = thresholding_technique
        assert isinstance(self.thresholding_technique, Thresholding)
        self.smooth_coeff_suppression = smooth_coeff_suppression
        assert isinstance(self.smooth_coeff_suppression, int)
        assert self.smooth_coeff_suppression >= -1
        self.artifact_filter_with_thresholded_XIC=artifact_filter_with_thresholded_XIC
        assert isinstance(self.artifact_filter_with_thresholded_XIC, bool)
        self.max_denoise = max_denoise
        assert isinstance(self.max_denoise, bool)

    def apply(self, intensities):
        """
        Given the extracted ion chromatogram intensities this function returns the denoised data via
        crane version 4
        """

        if np.count_nonzero(intensities) == 0:
            # Empty intensities matrix
            logging.debug('Empty intensities matrix')
            return intensities

        split_size = 2 ** self.max_level
        # mass_ranges class should ensure that strips are not too narrow
        assert intensities.shape[1] >= split_size / 2

        intensities = self.apply_transform(intensities, self.transform_technique)

        if self.split_data:
            # Split XIC and perform UDWT and calculate optimal wavelet decomposition level per split
            dwt_coeffs_splits, pad_lengths, split_dim, optimal_wavelet_level = self.adaptive_split_n_udwt(
                intensities,
                min_level=self.min_level,
                max_level=self.max_level,
                wavelet=self.wavelet,
                adaptive_level_selection_technique = self.adaptive_level_selection_technique,
            )
            num_splits = len(dwt_coeffs_splits)
        else:
            # perform UDWT
            dwt_coeffs, pad_len_col, pad_len_row, optimal_wavelet_level = self.adaptive_udwt(
                intensities,
                min_level=self.min_level,
                max_level=self.max_level,
                wavelet=self.wavelet,
                adaptive_level_selection_technique = self.adaptive_level_selection_technique,
            )
            dwt_coeffs_splits = [dwt_coeffs]
            num_splits = 1

        threshold_mode = 'hard'
        thresholded_coeffs = dwt_coeffs_splits.copy()

        for inx in range(num_splits):
            dwt_coeffs_split = dwt_coeffs_splits[inx]
            wavelet_levels = len(dwt_coeffs_split)

            # Estimate noise standard deviation
            (cH, cV, cD) = dwt_coeffs_split[-1][1]
            img_size = cD.shape[0] * cD.shape[1]
            visu_threshold, noise_sigma = self.cal_VisuShrink_threshold(cD, img_size)
            thresholded_coeff = dwt_coeffs_split.copy()

            # De-noise via selected thresholding technique
            for j in range(1, wavelet_levels + 1, 1):
                (cH, cV, cD) = dwt_coeffs_split[-j][1]
                threshold = self.calculate_threshold(cH, self.thresholding_technique, visu_threshold, noise_sigma, img_size)
                # When cHf is made zero for any level the performance degrades
                cHf = pywt.threshold(cH, threshold, mode=threshold_mode)
                threshold = self.calculate_threshold(cV, self.thresholding_technique, visu_threshold, noise_sigma, img_size)
                cVf = pywt.threshold(cV, threshold, mode=threshold_mode)
                threshold = self.calculate_threshold(cD, self.thresholding_technique, visu_threshold, noise_sigma, img_size)
                cDf = pywt.threshold(cD, threshold, mode=threshold_mode)
                if j <= self.smooth_coeff_suppression or self.smooth_coeff_suppression == -1:
                    thresholded_coeff[-j] = [np.zeros((cD.shape[0], cD.shape[1])), (cHf, cVf, cDf)]
                else:
                    thresholded_coeff[-j] = [dwt_coeffs_split[-j][0], (cHf, cVf, cDf)]
            thresholded_coeffs[inx] = thresholded_coeff
        
        thresholded_coeffs_copy = thresholded_coeffs.copy()
        
        if self.artifact_filter_with_thresholded_XIC or self.max_denoise:
            cal_xic_thresholded = True
        else:
            if self.apply_hrmc:
                cal_xic_thresholded = False
            else:
                cal_xic_thresholded = True

        if cal_xic_thresholded:
            if self.split_data:
                # Perform inverse UDWT and reconstruct data from splits
                xic_thresholded = self.adaptive_inverse_udwt_n_combine_splits(
                    coeffs=thresholded_coeffs_copy,
                    wavelet=self.wavelet,
                    pad_lengths=pad_lengths,
                    split_dim=split_dim,
                    optimal_wavelet_level=optimal_wavelet_level,
                    raw_data_shape=intensities.T.shape,
                )
            else:
                # perform inverse UDWT
                xic_thresholded = self.inverse_udwt(
                    coeffs=thresholded_coeffs_copy[0],
                    wavelet=self.wavelet,
                    pad_len_col=pad_len_col,
                    pad_len_row=pad_len_row,
                )
            xic_thresholded[xic_thresholded < 0] = 0
            mask_threshold = xic_thresholded > intensities.T
            xic_thresholded[mask_threshold] = intensities.T[mask_threshold]

        # horizontal row median correction (hrmc)
        if self.apply_hrmc:
            if self.split_data:
                thresholded_hrmc_coeffs = self.hrmc_splits(thresholded_coeffs)
                # Perform inverse UDWT and reconstruct data from splits
                xic_hrmc_denoised = self.adaptive_inverse_udwt_n_combine_splits(
                    coeffs=thresholded_hrmc_coeffs,
                    wavelet=self.wavelet,
                    pad_lengths=pad_lengths,
                    split_dim=split_dim,
                    optimal_wavelet_level=optimal_wavelet_level,
                    raw_data_shape=intensities.T.shape,
                )
            else:
                thresholded_hrmc_coeffs = [self.hrmc(thresholded_coeffs[0])]
                # perform inverse UDWT
                xic_hrmc_denoised = self.inverse_udwt(
                    coeffs=thresholded_hrmc_coeffs[0],
                    wavelet=self.wavelet,
                    pad_len_col=pad_len_col,
                    pad_len_row=pad_len_row,
                )
        else:
            thresholded_hrmc_coeffs = thresholded_coeffs.copy()
            xic_hrmc_denoised = xic_thresholded

        # Remove artefacts - Denoised has to be less than or equal to the original and non negative
        xic_hrmc_denoised[xic_hrmc_denoised < 0] = 0
        mask = xic_hrmc_denoised > intensities.T
        if self.artifact_filter_with_thresholded_XIC and self.apply_hrmc:
            xic_hrmc_denoised[mask] = xic_thresholded[mask]
        else:
            xic_hrmc_denoised[mask] = intensities.T[mask]

        xic_final_denoised = xic_hrmc_denoised.copy()
        
        if self.max_denoise:
            mask_final = xic_final_denoised > xic_thresholded
            xic_final_denoised[mask_final] = xic_thresholded[mask_final]

        xic_final_denoised = self.apply_inverse_transform(xic_final_denoised, self.transform_technique)

        return xic_final_denoised.T
    
    
class BEADS(DenoiserBase):
    """
    Denoise each single ion chromatogram acording to the algorithm described in "Ning, X., Selesnick, I.W. and Duval, L.
    Chromatogram baseline estimation and denoising using sparsity (BEADS). Chemometrics and Intelligent Laboratory Systems
    2014;139:156-167."
    """
    
    def __init__(
        self,
        d = 1,
        fc = 0.006,
        r = 6,
        Nit = 15,
        lam0 = 0.4, # lam0 = 0.5 * amp with amp = 0.8
        lam1 = 4.0, # lam1 = 5 * amp with amp = 0.8
        lam2 = 3.2, # lam2 = 4 * amp with amp = 0.8
        pen = 'L1_v2',
    ):
        """
        :param d int: Filter order (d = 1 or 2).
        :param fc float: Filter cut-off frequency (cycles/sample) (0 < fc < 0.5).
        :param r float: Asymmetry ratio for penalty function (r > 0).
        :param Nit int: Number of iteration (usually 10 to 30 is enough).
        :param lam0, lam1, lam2 float: Regularization parameters.
        :param pen string: Penalty function, 'L1_v1' or 'L1_v2'.
        """
        self.d = d
        assert isinstance(self.d, int), 'Filter order should be an integer'
        assert self.d > 0, 'Filter order should be positive'
        self.fc = fc
        assert isinstance(self.fc, float), 'Filter cut-off frequency should be a number'
        assert self.fc > 0, 'Filter cut-off frequency should be positive'
        assert self.fc < 0.5, 'Filter cut-off frequency should be less than 0.5'
        self.r = r
        assert self.r > 0, 'Asymmetry ratio for penalty function should be positive'
        self.Nit = Nit
        assert isinstance(self.Nit, int), 'Number of iteration should be an integer'
        assert self.Nit > 0, 'Number of iteration should be positive'
        self.lam0 = lam0
        assert isinstance(self.lam0, float), 'Regularization parameters should be numbers'
        assert self.lam0 > 0, 'Regularization parameters should be positive'
        self.lam1 = lam1
        assert isinstance(self.lam1, float), 'Regularization parameters should be numbers'
        assert self.lam1 > 0, 'Regularization parameters should be positive'
        self.lam2 = lam2
        assert isinstance(self.lam2, float), 'Regularization parameters should be numbers'
        assert self.lam2 > 0, 'Regularization parameters should be positive'
        self.pen = pen
        assert self.pen in ['L1_v1', 'L1_v2'], 'Penalty function should be either L1_v1 or L1_v2'
    
    def apply(self, intensities):
        """
        Given the extracted ion chromatogram intensities this function returns the denoised data via the algorithm 
        described in "Ning, X., Selesnick, I.W. and Duval, L. Chromatogram baseline estimation and denoising using sparsity
        (BEADS). Chemometrics and Intelligent Laboratory Systems 2014;139:156-167"
        """
        
        if np.count_nonzero(intensities) == 0:
            # Empty intensities matrix
            logging.debug('Empty intensities matrix')
            return intensities
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        denoised_intensities = np.empty(intensities.T.shape)
        xscale_l, xscale_r = 100, 100
        dx = 1
        
        for i in range(intensities.T.shape[0]):
            y = intensities.T[i,:]
            y_l = y[0]*sigmoid(1/xscale_l*np.arange(-5*xscale_l, 5*xscale_l, dx))
            y_r = y[-1]*sigmoid(-1/xscale_r*np.arange(-5*xscale_r, 5*xscale_r, dx))
            y_ext = np.hstack([y_l, y, y_r])
            len_l, len_o, len_r = len(y_l), len(y), len(y_r)
            signal_est, bg_est, cost = pybeads.beads(
                y_ext,
                self.d,
                self.fc,
                self.r,
                self.Nit,
                self.lam0,
                self.lam1,
                self.lam2,
                self.pen,
                conv=None
            )
            denoised_intensities[i,:] = signal_est[len_l: len_l+len_o]
            
        #Sometimes BEADS produce negative signal_est. Since MS data is positive we set all negative values to zero
        mask = denoised_intensities < 0
        denoised_intensities[mask] = 0
        
        return denoised_intensities.T