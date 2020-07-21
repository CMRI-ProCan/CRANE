import pandas as pd
import toffee


class SpectralLibrary():
    """
    SpectralLibrary data type.

    This is essentially just a wrapper around a pandas dataframes, `data`. It provides convinient inits from one
    file type, common operations and a standard format with which to pass to procantoolbox figure factories.
    """

    # This is the mass difference between the two isotopes of Carbon (C12 and C13)
    C13C12_MASS_DIFF = 1.0033548

    MINIMUM_HEADERS = [
        'PrecursorMz',
        'ProductMz',
        'PrecursorCharge',
        'ProductCharge',
        'LibraryIntensity',
        'NormalizedRetentionTime',
        'ProteinId',
        'PeptideSequence',
        'ModifiedPeptideSequence',
        'TransitionGroupId',
        'TransitionId',
    ]

    def __init__(self, df):
        """
        If headers need to be renamed, they will be. Furthermore, a ProductCharge will also be added with
        a default of 1 if it doesn't exist. This matches assumpitions made by OpenSwath.
        """
        consistent_headers = {
            'transition_group_id': 'TransitionGroupId',
            'transition_name': 'TransitionId',
            'FullUniModPeptideName': 'ModifiedPeptideSequence',
            'ProteinName': 'ProteinId',
            'Tr_recalibrated': 'NormalizedRetentionTime',
        }
        df = df.rename(columns=consistent_headers)

        if 'ProductCharge' not in df.columns:
            df['ProductCharge'] = 1

        missing_headers = set(self.MINIMUM_HEADERS).difference(set(df.columns))
        if len(missing_headers) > 0:
            raise RuntimeError('Missing important headers: {}'.format(missing_headers))
        self.data = df

    @classmethod
    def init_from_df(cls, df):
        return cls(df)

    @classmethod
    def init_from_file(cls, srl_fname):
        df = pd.read_table(srl_fname)
        return cls(df)

    def __eq__(self, other):
        return self.data.equals(other.data)

    def drop_duplicates(self, inplace=False):
        duplicate_cols = ['TransitionGroupId', 'ProductMz']
        df = self.data.drop_duplicates(subset=duplicate_cols)
        if inplace:
            self.data = df
            return self
        else:
            return self.init_from_df(df)

    def add_ms2_windows(self, swath_run=None, tof_fname=None, lower_offset=0.0, upper_offset=1.0):
        """
        Update the internal DataFrame with a column 'ms2Name' that maps the precursor m/z to the
        window in the the toffee file that is either represented by the swath_run, or the tof_fname
        """
        assert not (swath_run is None and tof_fname is None)
        assert not (swath_run is not None and tof_fname is not None)
        if swath_run is None:
            swath_run = toffee.SwathRun(tof_fname)
        assert isinstance(swath_run, toffee.SwathRun)

        precursor_ms2_name_map = swath_run.mapPrecursorsToMS2Names(
            self.data.PrecursorMz.unique(),
            lower_offset,
            upper_offset,
        )
        self.data['ms2Name'] = self.data.PrecursorMz.map(precursor_ms2_name_map)
        self.data = self.data[~self.data.ms2Name.isna()]

    def calculate_isotope_mz(self, n_isotopes=4, drop_other_cols=True, sort_by_intensity=False):
        """
        Calculates the isotope masses for both precursors and products in the spectral library
        by assuming that the mass offset is derived from the mass difference between Carbon12 and
        Carbon13

        :param int n_isotopes: a positive integer that determines how many isotope masses will be
            included in the output. Specifically, the returned data frames will include masses of
            `[ion_a, ion_a + 1 * offset_a, ..., ion_a + n_isotopes * offset_a,
            ion_b, ion_b + 1 * offse_b, ..., ion_b + n_isotopes * offset_b, ...]` where `offset_i` is
            related to the difference between the masses of Carbon12 and Carbon13 and the charge of
            `ion_i`.
        :param bool drop_other_cols: if True, other columns of the SpectralLibrary will be excluded
            from the returned data frames
        :param bool sort_by_intensity: if True, the results will be ordered by the LibraryIntensity

        :return: a pair of :class:`pandas.DataFrame` representing the precusor and product isotope
            offsets. The dataframes have Id and Charge coloumns that match the data in the spectral
            library, as well as IsotopeNr and IsotopeMz that represent the number of the isotope (i.e.
            its distance from the monoisotopic m/z) and the isotope m/z, respectively.
        """
        assert n_isotopes >= 0

        def _impl(id_col, mz_col, charge_col, drop_duplicates, grouping_col=None):
            intensity_col = 'LibraryIntensity'
            # set up the data to build the isotopes
            cols = [id_col, mz_col, charge_col]
            if grouping_col is not None:
                cols.append(grouping_col)
            if sort_by_intensity:
                cols.append(intensity_col)
            if drop_other_cols:
                df = self.data.loc[:, cols].copy()
            else:
                df = self.data.copy()
            if drop_duplicates:
                df = df.drop_duplicates(subset=[id_col, mz_col, charge_col])
            df['IsotopeNr'] = 0
            col_names = {id_col: 'Id', charge_col: 'Charge', mz_col: 'IsotopeMz'}
            if grouping_col is not None:
                col_names[grouping_col] = 'GroupId'
            df.rename(columns=col_names, inplace=True)

            # fill in the isotope number
            isotopes = []
            for i in range(n_isotopes + 1):
                df['IsotopeNr'] = i
                isotopes.append(df.copy())
            isotopes = pd.concat(isotopes)

            # calculate the m/z offsets for each isotope but assuming that it is offset
            # by a multiple of the C12/C13 mass difference
            isotopes['IsotopeMz'] += isotopes.IsotopeNr * self.C13C12_MASS_DIFF / isotopes.Charge

            # sort the result
            sort_cols = []
            ascending = []
            if grouping_col is None:
                sort_cols.append('Id')
                ascending.append(True)
            else:
                sort_cols.append('GroupId')
                ascending.append(True)
            if sort_by_intensity:
                sort_cols.append(intensity_col)
                ascending.append(False)
            if grouping_col is not None:
                sort_cols.append('Id')
                ascending.append(True)
            sort_cols.append('IsotopeNr')
            ascending.append(True)
            isotopes.sort_values(sort_cols, ascending=ascending, inplace=True)

            return isotopes

        # for the precursors, we need to drop duplicates, as we only want one
        # set per transition group
        precursor_rows = _impl(
            id_col='TransitionGroupId',
            mz_col='PrecursorMz',
            charge_col='PrecursorCharge',
            drop_duplicates=True,
        )

        # for the product ions, we assume that these are unique in the SpectralLibrary
        product_rows = _impl(
            id_col='TransitionId',
            mz_col='ProductMz',
            charge_col='ProductCharge',
            drop_duplicates=True,
            grouping_col='TransitionGroupId',
        )

        return precursor_rows, product_rows
