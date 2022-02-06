class CollectionAttributes(object):
    LSCI_LED_LAMBDAS = [1310]
    SWIR_LED_LAMBDAS = [1200, 1300, 1450, 1550]
    BI_LED_LAMBDAS = [940]

    # The canonical size is the one with respect to which PA ROI's are defined.
    CANONICAL_FINGER_SIZE = {'width': 21, 'height': 61}

    # Mapping from SWIR to the canonical size and view
    SWIR_FINGER_SLIT_TO_CANONICAL = {
        'crop': {'x1': 21, 'x2': 42, 'y1': 1, 'y2': 62},
        'rot90': 0,
        'fliplr': False
    }

    CANONICAL_FINGER_ROI = {
        'x1': 4,
        'x2': 18,
        'y1': 25,
        'y2': 51
    }

    def get_dictionary_attribute_value(self, data_identifier: str, attribute_dict_name: str,
                                       check_for_key: bool = True):
        assert isinstance(data_identifier, str), \
            "{} Error: Variable 'data_identifier' must be a " \
            "string.".format(self.__class__.__name__ + ".get_dictionary_attribute_value()")

        assert isinstance(attribute_dict_name, str), \
            "{} Error: Variable 'attribute_dict_name' must be a " \
            "string.".format(self.__class__.__name__ + ".get_dictionary_attribute_value()")

        assert isinstance(check_for_key, bool), \
            "{} Error: Variable 'check_for_key' must be " \
            "boolean.".format(self.__class__.__name__ + ".get_dictionary_attribute_value()")

        assert hasattr(self, attribute_dict_name), \
            "{} Error: dataset '{}' does not support '{}' " \
            "property.".format(self.__class__.__name__ + ".get_dictionary_attribute_value()",
                               data_identifier, attribute_dict_name)

        attribute_dict = getattr(self, attribute_dict_name)

        if check_for_key:
            assert data_identifier in attribute_dict.keys(), \
                "{} Error: dataset does not contain '{}' data.".format(self.__class__.__name__, data_identifier)

        if data_identifier in attribute_dict.keys():
            return attribute_dict[data_identifier]
        else:
            return None

    @staticmethod
    def _get_dataset_name_mapping(data_identifier: str, dataset_names: list):
        return [data_identifier + '_' + str(i) for i in range(len(dataset_names))]

    def get_data_bit_depth(self, data_identifier: str) -> int:
        return self.get_dictionary_attribute_value(data_identifier, 'BIT_DEPTH_DICT')

    def get_data_normalization_factor(self, data_identifier: str) -> (int, float):
        return self.get_dictionary_attribute_value(data_identifier, 'NORM_FACTOR_DICT')

    def get_data_image_transform(self, data_identifier: str) -> (None, list):
        return self.get_dictionary_attribute_value(data_identifier, 'IMAGE_TRANSFORM_DICT', False)

    @staticmethod
    def choose_and_validate_wavelengths(default_wavelengths: list, wavelengths_index: (None, list) = None) \
            -> (None, list):
        if wavelengths_index is not None:
            assert isinstance(wavelengths_index, list) and len(wavelengths_index) > 0 and all(
                list(map(lambda item: isinstance(item, int) and item >= 0, wavelengths_index))), \
                "{} Error: Variable 'wavelengths' must be a list of non-negative " \
                "integers.".format('CollectionAttributes.choose_and_validate_wavelengths()')

            assert max(wavelengths_index) < len(default_wavelengths), \
                "{} Error: Variable 'wavelengths_index' contains numbers that are out of bounds. " \
                "There are '{}' available wavelengths " \
                "[{}]".format('CollectionAttributes.choose_and_validate_wavelengths()', len(default_wavelengths),
                              ', '.join(["'" + str(wavelength) + "'" for wavelength in default_wavelengths]))

            return [default_wavelengths[i] for i in wavelengths_index]
        else:
            return default_wavelengths


class USC1CollectionAttributes(CollectionAttributes):
    BASLER_DS_NAME_PREFIX = '/BASLER_'
    SWIR_DS_NAME_PREFIX = '/BOBCAT_SWIR_'
    LSCI_DS_NAME_PREFIX = '/BOBCAT_LSCI'
    VIS_LED_LAMBDAS_FINGER = [465, 591, 0]
    NIR_LED_LAMBDAS_FINGER = [720, 780, 870, 940]

    # Mapping from multi-spectral NIR/Visible and back-illumination to the canonical size and view
    VIS_FINGER_SLIT_TO_CANONICAL = NIR_FINGER_SLIT_TO_CANONICAL = \
        {
            'crop': {'x1': 5, 'x2': 1282, 'y1': 35, 'y2': 485},
            'rot90': -1,
            'fliplr': True
        }

    # Mapping from SWIR and LSCI to the canonical size and view
    SWIR_FINGER_SLIT_TO_CANONICAL = LSCI_FINGER_SLIT_TO_CANONICAL = \
        {
            'crop': {'x1': 0, 'x2': 320, 'y1': 51, 'y2': 171},
            'rot90': -1,
            'fliplr': True
        }

    BIT_DEPTH_DICT = {'FINGER_READER_LS': 16,
                      'FINGER_READER_VNB': 12}

    @classmethod
    def basler_datasets(cls, lit_prefix, lambdas, lit_suffix):
        cam_prefix = cls.BASLER_DS_NAME_PREFIX
        basler_ds = [cam_prefix + lit_prefix + ("white" if b == 0 else str(b) + "nm") + lit_suffix for b in lambdas]
        return basler_ds

    @classmethod
    def vis_finger_datasets(cls, lighting):
        return cls.basler_datasets("VIS_", cls.VIS_LED_LAMBDAS_FINGER, lighting)

    @classmethod
    def vis_light_finger_datasets(cls):
        return cls.vis_finger_datasets('')

    @classmethod
    def vis_dark_finger_datasets(cls):
        return cls.vis_finger_datasets('_Dark')

    @classmethod
    def nir_finger_datasets(cls, lighting):
        return cls.basler_datasets("NIR_", cls.NIR_LED_LAMBDAS_FINGER, lighting)

    @classmethod
    def nir_light_finger_datasets(cls):
        return cls.nir_finger_datasets('')

    @classmethod
    def nir_dark_finger_datasets(cls):
        nir_dark_ds = cls.nir_finger_datasets('_Dark')
        ''' Due to a configuration error there is no dark image for wavelength 940nm in USC1. Here we simply replace it
        with the dark image for the 870nm wavelength.
        '''
        return [ds.replace("940", "870") for ds in nir_dark_ds]

    @classmethod
    def swir_datasets(cls, lighting):
        return [cls.SWIR_DS_NAME_PREFIX + str(wl) + "nm" + lighting for wl in cls.SWIR_LED_LAMBDAS]

    @classmethod
    def swir_light_datasets(cls):
        return cls.swir_datasets('')

    @classmethod
    def swir_dark_datasets(cls):
        swir_dark_ds = cls.swir_datasets('_Dark')
        ''' Due to a configuration error there is no dark images for all wavelengths but 1200nm in USC1. "
        "Here we simply replace them with the dark image for the 1200nm wavelength.
        '''
        return [ds.replace("1300", "1200").replace("1450", "1200").replace("1550", "1200") for ds in swir_dark_ds]

    @classmethod
    def lsci_datasets(cls, lighting):
        return [cls.LSCI_DS_NAME_PREFIX + lighting]

    @classmethod
    def lsci_light_datasets(cls):
        return cls.lsci_datasets('')

    @classmethod
    def lsci_dark_datasets(cls):
        """ There are no dark images for LSCI data starting ST3.
        Dark images for the first available SWIR channel will be returned instead
        """
        lsci_dark = []
        swir_dark = cls.swir_dark_datasets()
        if len(swir_dark) > 0:
            lsci_dark = [swir_dark[0]]
        return lsci_dark

    @classmethod
    def bi_light_finger_datasets(cls):
        return cls.basler_datasets("BI_", cls.BI_LED_LAMBDAS, "")

    @classmethod
    def bi_dark_finger_datasets(cls):
        return []


class USC2CollectionAttributes(USC1CollectionAttributes):

    # Mapping from multi-spectral NIR/Visible and back-illumination to the canonical size and view
    VIS_FINGER_SLIT_TO_CANONICAL = NIR_FINGER_SLIT_TO_CANONICAL = \
        {
            'crop': {'x1': 6, 'x2': 1282, 'y1': 25, 'y2': 490},
            'rot90': -1,
            'fliplr': True
        }

    # Mapping from SWIR and LSCI to the canonical size and view
    SWIR_FINGER_SLIT_TO_CANONICAL = LSCI_FINGER_SLIT_TO_CANONICAL = {
        'crop': {'x1': 0, 'x2': 320, 'y1': 70, 'y2': 182},
        'rot90': -1,
        'fliplr': True
    }

    @classmethod
    def nir_dark_finger_datasets(cls):
        return cls.nir_finger_datasets('_Dark')

    @classmethod
    def swir_dark_datasets(cls):
        return cls.swir_datasets('_Dark')


def get_collection_attribute_set(collection_id: (None, str)):
    if collection_id is not None:
        assert isinstance(collection_id, str), "Variable 'collection_id' must be a string."
    else:
        return CollectionAttributes()

    if collection_id == 'USC1':
        return USC1CollectionAttributes()
    elif collection_id == 'USC2':
        return USC2CollectionAttributes()


def guess_collection_attribute_set(file):
    collection_id = None
    try:
        collection_id = file.get_collection_id()
    except Exception as e:
        print("'guess_collection_attribute_set()' Error: {}".format(str(e)))
        pass

    if collection_id is not None:
        return get_collection_attribute_set(collection_id)
