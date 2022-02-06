from typing import Callable

import numpy as np

from padisi_modules.dataIO.padisi_dataset import PadisiH5Reader, PadisiH5ReaderDataExtractor


class FacePadisiPreprocessedDataExtractor(PadisiH5ReaderDataExtractor):
    NEEDED_DICT_KEYS = ('data', 'identifiers')
    EXTRACTOR_NAME_FIXED_ORDER = ('COLOR',)
    EXTRACTOR_CHANNEL_MAPPING = {'COLOR': tuple(range(3))}

    def __init__(self, data_identifiers: (tuple, list), preprocessed_data_dict: dict):
        super(FacePadisiPreprocessedDataExtractor, self).__init__()
        self.preprocessed_data_dict = preprocessed_data_dict
        self.data_identifiers = data_identifiers
        self._data_channels = sorted([item for sublist in
                                      [FacePadisiPreprocessedDataExtractor.EXTRACTOR_CHANNEL_MAPPING[key]
                                       for key in self.data_identifiers] for item in sublist])
        # Pre-calculate indices per identifier for speed
        self._identifier_idx = dict(zip(self.preprocessed_data_dict['identifiers'],
                                        range(len(self.preprocessed_data_dict['identifiers']))))

    def _extract_padisi_h5_data(self, padisi_h5_reader: PadisiH5Reader) -> np.ndarray:
        reader_identifier = padisi_h5_reader.get_reader_identifier()
        assert reader_identifier in self._identifier_idx, \
            "{} Error: Identifier '{}' does not exist in " \
            "dataset.".format(self.__class__.__name__ + "._extract_padisi_h5_data()", reader_identifier)

        data = self.preprocessed_data_dict['data'][self._identifier_idx[reader_identifier], self._data_channels, :, :]
        if data.ndim < 4:
            data = np.expand_dims(data, axis=0)

        return data

    def get_property_extractors_validation_dictionary(self) -> (None, Callable, dict):
        pass

    @property
    def data_identifiers(self) -> tuple:
        return self._data_identifiers

    @data_identifiers.setter
    def data_identifiers(self, data_identifiers: (tuple, list)) -> None:
        assert isinstance(data_identifiers, (tuple, list)) and len(data_identifiers) > 0 and all(
            list(map(lambda item: item in FacePadisiPreprocessedDataExtractor.EXTRACTOR_CHANNEL_MAPPING,
                     data_identifiers))) and len(set(data_identifiers)) == len(data_identifiers), \
            "{} Error: Property 'data_identifiers' must be a non-empty tuple or list containing unique strings: " \
            "[{}].".format(self.__class__.__name__,
                           ', '.join(["'" + item + "'" for item in
                                      FacePadisiPreprocessedDataExtractor.EXTRACTOR_CHANNEL_MAPPING]))
        if isinstance(data_identifiers, list):
            data_identifiers = tuple(data_identifiers)
        self._data_identifiers = data_identifiers

    @property
    def preprocessed_data_dict(self) -> dict:
        return self._preprocessed_data_dict

    @preprocessed_data_dict.setter
    def preprocessed_data_dict(self, preprocessed_data_dict: dict) -> None:
        assert isinstance(preprocessed_data_dict, dict) and len(preprocessed_data_dict) > 0, \
            "{} Error: Property 'preprocessed_data_dict' must be a non-empty " \
            "dictionary.".format(self.__class__.__name__)
        missing_keys = [item for item in FacePadisiPreprocessedDataExtractor.NEEDED_DICT_KEYS
                        if item not in preprocessed_data_dict]
        assert len(missing_keys) == 0, \
            "{} Error: Dictionary 'preprocessed_data_dict' is missing keys: " \
            "[{}].".format(self.__class__.__name__, ', '.join(["'" + item + "'" for item in missing_keys]))

        assert isinstance(preprocessed_data_dict['data'], np.ndarray) and preprocessed_data_dict['data'].ndim == 4, \
            "{} Error: Entry 'preprocessed_data_dict['data']' must be a 4-dimensional numpy " \
            "array.".format(self.__class__.__name__)

        assert isinstance(preprocessed_data_dict['identifiers'], list) and len(
            preprocessed_data_dict['identifiers']) and all(
            list(map(lambda item: isinstance(item, str) and len(item) > 0, preprocessed_data_dict['identifiers']))), \
            "{} Error: Entry 'preprocessed_data_dict['identifiers']' must be a non-empty list of " \
            "non-empty strings.".format(self.__class__.__name__)

        assert len(preprocessed_data_dict['identifiers']) == preprocessed_data_dict['data'].shape[0], \
            "{} Error: The number of elements of entry 'preprocessed_data_dict['identifiers']' and the fist " \
            "dimension of the numpy array 'preprocessed_data_dict['data']' must match".format(self.__class__.__name__)

        self._preprocessed_data_dict = preprocessed_data_dict
