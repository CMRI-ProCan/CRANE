import numpy as np
import toffee


class MassRangeCalculatorBase():
    """
    Abstract base class
    """

    def calculate_ranges(self, swath_map, mz_list, **kwargs):
        raise NotImplementedError('Called from base class')

    def complement_mass_ranges(self, swath_map, mz_ranges):
        return list()


class MassRangeCalculatorFullMap(MassRangeCalculatorBase):
    """
    Calcualtes a single range that encompasses the full swath map
    """

    def __init__(self, upper_offset=0, lower_offset=0):
        """
        :param int upper_offset: allow an offset to be taken from the upper edge of the
            swath map -- this is really only useful for testing
        :param int lower_offset: allow an offset to be taken from the lower edge of the
            swath map -- this is really only useful for testing
        """
        self.upper_offset = upper_offset
        self.lower_offset = lower_offset

    def calculate_ranges(self, swath_map, mz_list):
        mz_transformer = swath_map.getMzTransformer()
        lower = mz_transformer.lowerMzInIMSCoords() + self.lower_offset
        upper = mz_transformer.upperMzInIMSCoords() - self.upper_offset
        assert upper >= lower
        return [toffee.MassOverChargeRangeIMSCoords(lower, upper)]


class MassRangeCalculatorFullMapSplit(MassRangeCalculatorBase):
    """
    Calculates slices through the full mass over charge space such
    that each slice is touching.

    Use this class if you wish to denoise a full swath window without
    sub-sampling
    """

    def __init__(self, split_size=128):
        self.split_size = split_size

    def calculate_ranges(self, swath_map, mz_list):
        mz_transformer = swath_map.getMzTransformer()
        lower = mz_transformer.lowerMzInIMSCoords()
        max_upper = mz_transformer.upperMzInIMSCoords()
        ranges = list()
        while lower < max_upper:
            if lower + self.split_size - 1 <= max_upper:
                upper = lower + self.split_size - 1
            else:
                upper = max_upper
                lower = upper - self.split_size + 1
            ranges.append(
                toffee.MassOverChargeRangeIMSCoords(
                    lower,
                    upper,
                )
            )
            lower = upper + 1  # these ranges are inclusive
        return ranges


class MassRangeCalculatorNonOverlappingBase(MassRangeCalculatorBase):
    """
    Abstract base class that takes a list of calculated ranges and converts them
    so that they are not overlapping
    """

    def _convert_to_non_overlapping_regions(self, swath_map, mz_ranges):
        full_mz = swath_map.massOverCharge()
        transformer = swath_map.getMzTransformer()
        base_mass_over_charge_offset = transformer.toIMSCoords(full_mz[0])

        # calculate a mask of non-overlapping regions. Zero in the mask means
        # outside, one means inside.
        mask = np.zeros(full_mz.shape)
        for mz_range in mz_ranges:
            assert isinstance(mz_range, toffee.IMassOverChargeRange)
            lower = mz_range.lowerMzInIMSCoords(transformer) - base_mass_over_charge_offset
            upper = mz_range.upperMzInIMSCoords(transformer) - base_mass_over_charge_offset + 1
            mask[lower:upper] = 1

        # convert mask to non-overlapping pairs
        range_pairs = list()
        last_mask = mask[0]
        left = None if last_mask == 0.0 else 0
        right = None
        for idx, cur_mask in enumerate(mask[1:]):
            idx = idx + 1
            if last_mask == 0 and cur_mask == 1:
                assert left is None
                assert right is None
                left = idx
            elif last_mask == 1 and cur_mask == 0:
                assert left is not None
                assert right is None
                right = idx - 1  # minus 1 as the ranges are inclusive
                range_pairs.append((left, right))
                left, right = None, None
            last_mask = cur_mask

        # convert range pairs into toffee ranges
        non_overlapping_ranges = list()
        window_upper_limit = transformer.upperMzInIMSCoords() - base_mass_over_charge_offset + 1
        window_lower_limit = transformer.lowerMzInIMSCoords() - base_mass_over_charge_offset + 1
        for (left, right) in range_pairs:
            # Ensure mz ranges are withing split_size
            lower = left
            max_upper = min(right, window_upper_limit)
            while lower < max_upper:
                upper = min(lower + self.split_size - 1, max_upper)
                non_overlapping_ranges.append(
                    toffee.MassOverChargeRangeIMSCoords(
                        lower + base_mass_over_charge_offset,
                        upper + base_mass_over_charge_offset,
                    )
                )
                lower = upper + 1  # these ranges are inclusive

            # Ensure all mz ranges have sufficient pixels for denoising after spliting
            # For small strips, merge with adjacent strip if available, extend if not.
            remainder = (upper - left + 1) % self.split_size
            if remainder != 0 and remainder < self.split_size / 2:
                if len(non_overlapping_ranges) >= 2:
                    lower_idx_latest = non_overlapping_ranges[-1].lowerMzInIMSCoords(transformer)
                    upper_idx_latest = non_overlapping_ranges[-1].upperMzInIMSCoords(transformer)
                    lower_idx_preceding = non_overlapping_ranges[-2].lowerMzInIMSCoords(transformer)
                    upper_idx_preceding = non_overlapping_ranges[-2].upperMzInIMSCoords(transformer)
                    # Check for adjacent strip
                    if lower_idx_latest - 1 == upper_idx_preceding:
                        upper = upper_idx_latest
                        lower = lower_idx_preceding
                        non_overlapping_ranges.pop(-1)
                    else:
                        upper = upper_idx_latest
                        lower = max(upper - self.split_size + 1, window_lower_limit)
                        # Check if the extention of the lower limit causes an overlap
                        if lower <= upper_idx_preceding:
                            lower = lower_idx_preceding
                            non_overlapping_ranges.pop(-1)
                else:
                    upper = non_overlapping_ranges[-1].upperMzInIMSCoords(transformer)
                    lower = max(upper - self.split_size + 1, window_lower_limit)

                non_overlapping_ranges[-1] = toffee.MassOverChargeRangeIMSCoords(
                                                lower,
                                                upper,
                                            )
        return non_overlapping_ranges

    def _calculate_complement_mass_ranges(self, swath_map, mz_ranges):
        full_mz = swath_map.massOverCharge()
        transformer = swath_map.getMzTransformer()
        base_mass_over_charge_offset = transformer.toIMSCoords(full_mz[0])

        # calculate a mask of non-overlapping regions. Zero in the mask means
        # outside, one means inside.
        mask = np.zeros(full_mz.shape)
        for mz_range in mz_ranges:
            assert isinstance(mz_range, toffee.IMassOverChargeRange)
            lower = mz_range.lowerMzInIMSCoords(transformer) - base_mass_over_charge_offset
            upper = mz_range.upperMzInIMSCoords(transformer) - base_mass_over_charge_offset + 1
            mask[lower:upper] = 1

        # calculate complement mass ranges from mask
        range_pairs = list()
        last_mask = mask[0]
        left = None if last_mask == 1 else 0
        right = None
        for idx, cur_mask in enumerate(mask[1:]):
            idx = idx + 1
            if last_mask == 1 and cur_mask == 0:
                assert left is None
                assert right is None
                left = idx
            elif last_mask == 0 and cur_mask == 1:
                assert left is not None
                assert right is None
                right = idx - 1  # minus 1 as the ranges are inclusive
                range_pairs.append((left, right))
                left, right = None, None
            last_mask = cur_mask

        if left is not None:
            assert right is None
            right = idx
            range_pairs.append((left, right))

        # convert range pairs into mz ranges
        complement_mass_ranges = list()
        for (left, right) in range_pairs:
            complement_mass_ranges.append(
                toffee.MassOverChargeRangeIMSCoords(
                    left + base_mass_over_charge_offset,
                    right + base_mass_over_charge_offset,
                )
            )
        return complement_mass_ranges


class MassRangeCalculatorNonOverlappingPPMFullWidth(MassRangeCalculatorNonOverlappingBase):
    """
    Take slices through the mass range, where the width of the slice is set by
    a ppm width
    """

    def __init__(self, ppm_full_width=100, split_size=128):
        self.ppm_full_width = ppm_full_width
        self.split_size = split_size

    def calculate_ranges(self, swath_map, mz_list):
        ranges = list()
        for mz in mz_list:
            ranges.append(
                toffee.MassOverChargeRangeWithPPMFullWidth(
                    mz,
                    self.ppm_full_width,
                )
            )
        return self._convert_to_non_overlapping_regions(swath_map, ranges)

    def complement_mass_ranges(self, swath_map, mz_ranges):
        return self._calculate_complement_mass_ranges(swath_map, mz_ranges)


class MassRangeCalculatorNonOverlappingPixelHalfWidth(MassRangeCalculatorNonOverlappingBase):
    """
    Take slices through the mass range, where the width of the slice is set by
    a pixel width
    """

    def __init__(self, px_half_width=35, split_size=128):
        self.px_half_width = px_half_width
        self.split_size = split_size

    def calculate_ranges(self, swath_map, mz_list):
        ranges = list()
        for mz in mz_list:
            ranges.append(
                toffee.MassOverChargeRangeWithPixelHalfWidth(
                    mz,
                    self.px_half_width,
                )
            )
        return self._convert_to_non_overlapping_regions(swath_map, ranges)

    def complement_mass_ranges(self, swath_map, mz_ranges):
        return self._calculate_complement_mass_ranges(swath_map, mz_ranges)
