from padisi_modules.dataIO.dataset import ReaderDataExtractor
from padisi_modules.dataIO.padisi_dataset import PadisiH5Dataset

from padisi_modules.dataset_utils.face_datasets.padisi_face_datasets_utils import FacePadisiPreprocessedDataExtractor

VALID_PARTS = ('COLOR',)


def create_padisi_face_data_extractor(extractor_id: str,
                                      preprocessed_data_dict: (dict, None) = None) -> ReaderDataExtractor:

    error_str = 'create_padisi_face_data_extractor()'
    assert preprocessed_data_dict is not None, \
        "{} Error: Using non-preprocessed data is currently not supported.".format(error_str)

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
        return FacePadisiPreprocessedDataExtractor(extractor_id_parts, preprocessed_data_dict)


def create_padisi_face_dataset(extractor_id: str, db_path: (None, str), gt_path: str,
                               part_path: str, multi_label: bool = False,
                               pai_code_label_mapping: (None, dict) = None,
                               check_file_existence: bool = False,
                               preprocessed_data_dict: (dict, None) = None) -> PadisiH5Dataset:
    """
    This function is used to create a dataset object for a particular algorithm.

    :param extractor_id          : String for extractor_id to be used. See create_face_data_extractor above.
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

    extractor = create_padisi_face_data_extractor(extractor_id, preprocessed_data_dict)
    dataset = PadisiH5Dataset(db_path=db_path, ground_truth_path=gt_path, dataset_partition_path=part_path,
                              modality='FACE', data_extractor=extractor,
                              multi_label=multi_label, pai_code_label_mapping=pai_code_label_mapping,
                              check_file_existence=check_file_existence)
    return dataset
