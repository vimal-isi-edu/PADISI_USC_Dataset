from padisi_modules.dataIO.dataset import ReaderDataExtractor
from padisi_modules.dataIO.padisi_dataset import PadisiH5Dataset
from padisi_modules.dataset_utils.finger_datasets.padisi_finger_datasets_utils import \
    FingerSingleSpecRangePadisiH5ReaderDataExtractor, FingerMultiSpecRangePadisiH5ReaderDataExtractor, \
    FingerPadisiPreprocessedDataExtractor, spec_range_extract_attrs

VALID_PARTS = ('FM', 'FS', 'FL', 'BN')


def create_padisi_finger_data_extractor(extractor_id: str,
                                        preprocessed_data_dict: (dict, None) = None) -> ReaderDataExtractor:

    if extractor_id.startswith('Full'):
        assert preprocessed_data_dict is None, \
            "{} Error: If 'extractor_id' starts with 'Full' the raw .h5 file data need to be " \
            "used.".format('create_padisi_finger_data_extractor()')
        # Extract full Visible and NIR images
        if extractor_id == 'FullFM':
            return FingerMultiSpecRangePadisiH5ReaderDataExtractor(
                [
                    spec_range_extract_attrs(name='vis'),
                    spec_range_extract_attrs(name='nir'),
                ], concat_axis=1, mode='diff', crop_roi=False, auto_canon_target=False, bit_depth_norm=True)
        # Extract full SWIR images
        elif extractor_id == 'FullFS':
            return FingerSingleSpecRangePadisiH5ReaderDataExtractor('swir', crop_roi=False, auto_canon_target=False,
                                                                    mode='diff', bit_depth_norm=True)
        # Extract full LSCI images (10 frames out of 100 - from 10 to 19)
        elif extractor_id == 'FullFL':
            return FingerSingleSpecRangePadisiH5ReaderDataExtractor('lsci', crop_roi=False, auto_canon_target=False,
                                                                    mode='diff', lit_frames=list(range(10, 20)),
                                                                    bit_depth_norm=True)
        # Extract full Back-Illumination images (3 frames out of 20 - from 10 to 12)
        elif extractor_id == 'FullBN':
            return FingerSingleSpecRangePadisiH5ReaderDataExtractor('bi', crop_roi=False, auto_canon_target=False,
                                                                    mode='diff', lit_frames=list(range(10, 13)),
                                                                    bit_depth_norm=True)
        else:
            raise RuntimeError("{} Error: If 'extractor_id' starts with 'Full', it must be one of "
                               "['FullFM', 'FullFS', 'FullFL', "
                               "'FullBN'].".format('create_padisi_finger_data_extractor()'))

    # Extractor is of the form FM_FS_FL_BN (any combination is acceptable, e.g. FM, FM_FS, FM_FL, FM_FL_BN, etc.)
    # Data is always reordered with priority FL -> FS -> FM -> BN. For example (if FS_FL is provided, the actual
    # order of the data will be FL -> FS)
    extractor_id_parts = extractor_id.split('_')
    assert len(set(extractor_id_parts)) == len(extractor_id_parts), \
        "{} Error: Variable 'extractor_id' must contain unique parts.".format('create_padisi_finger_data_extractor()')

    assert all(list(map(lambda item: item in VALID_PARTS, extractor_id_parts))), \
        "{} Error: Variable 'extractor_id' must contain strings [{}], separated by " \
        "underscores.".format('create_padisi_finger_data_extractor()',
                              ', '.join(["'" + item + "'" for item in VALID_PARTS]))

    if preprocessed_data_dict is not None:
        assert isinstance(preprocessed_data_dict, dict) and len(preprocessed_data_dict) > 0, \
            "{} Error: If variable 'preprocessed_data_dict' is not None, it must be non-empty " \
            "dictionary.".format('create_padisi_finger_data_extractor()')
        return FingerPadisiPreprocessedDataExtractor(extractor_id_parts, preprocessed_data_dict)
    else:
        spec_range_extract_attrs_list = []
        if 'FM' in extractor_id_parts:
            spec_range_extract_attrs_list.extend([
                spec_range_extract_attrs(name='vis', roi_relative_to='target',
                                         roi={'x1': 20, 'x2': 100, 'y1': 140, 'y2': 300}),
                spec_range_extract_attrs(name='nir', roi_relative_to='target',
                                         roi={'x1': 20, 'x2': 100, 'y1': 140, 'y2': 300})])
        if 'FS' in extractor_id_parts:
            spec_range_extract_attrs_list.append(
                spec_range_extract_attrs(name='swir', roi_relative_to='target',
                                         roi={'x1': 20, 'x2': 100, 'y1': 140, 'y2': 300}))
        if 'FL' in extractor_id_parts:
            spec_range_extract_attrs_list.append(
                spec_range_extract_attrs(name='lsci', lit_frames=list(range(10, 20)),
                                         roi_relative_to='target', roi={'x1': 20, 'x2': 100, 'y1': 140, 'y2': 300}))
        if 'BN' in extractor_id_parts:
            spec_range_extract_attrs_list.append(
                spec_range_extract_attrs(name='bi', lit_frames=list(range(10, 13, 1)), roi_relative_to='target',
                                         roi={'x1': 20, 'x2': 100, 'y1': 140, 'y2': 300}))
        return FingerMultiSpecRangePadisiH5ReaderDataExtractor(
            spec_range_extract_attrs_list, concat_axis=1,
            target_slit_size={'width': 120, 'height': 320}, bit_depth_norm=True)


def create_padisi_finger_dataset(extractor_id: str, db_path: (None, str), gt_path: str,
                                 part_path: str, multi_label: bool = False,
                                 pai_code_label_mapping: (None, dict) = None,
                                 check_file_existence: bool = False,
                                 preprocessed_data_dict: (dict, None) = None) -> PadisiH5Dataset:
    """
    This function is used to create a dataset object for a particular algorithm.

    :param extractor_id          : String for extractor_id to be used. See create_finger_data_extractor above.
    :param db_path               : path to database or None.
    :param gt_path               : ground truth path.
    :param part_path             : partition file's path, typically left None in deployment.
    :param multi_label           : See PadisiH5Dataset for more details.
                       (Optional)  Default is False.
    :param pai_code_label_mapping: See PadisiH5Dataset for more details.
                       (Optional)  Default is None.
    :param check_file_existence  : See PadisiH5Dataset for more details.
                       (Optional)  Default is False.
    :param preprocessed_data_dict: Data dictionary of pre-processed data (using the provided preprocessed data file)
                       (Optional)  Default is None.
    :return                      : Dataset object configured with the proper extractor
    """

    assert isinstance(extractor_id, str) and len(extractor_id) > 0, \
        "{} Error: Variable 'extractor_id' must be a non-empty string.".format('create_padisi_finger_dataset()')

    extractor = create_padisi_finger_data_extractor(extractor_id, preprocessed_data_dict)
    dataset = PadisiH5Dataset(db_path=db_path, ground_truth_path=gt_path, dataset_partition_path=part_path,
                              modality='FINGER', data_extractor=extractor,
                              multi_label=multi_label, pai_code_label_mapping=pai_code_label_mapping,
                              check_file_existence=check_file_existence)
    return dataset
