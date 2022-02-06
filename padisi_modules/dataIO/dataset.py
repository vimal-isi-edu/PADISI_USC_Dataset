import csv
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable


# Base class DataReader for reading data from different files

# Abstract methods:
#    .get_data()
#    .get_reader_info_dict()
#    .get_reader_info_str()
#    .get_reader_identifier()

# Available methods:
#    .get_ground_truth()
class DataReader(ABC):
    def __init__(self, ground_truth):
        """
        :param ground_truth: Ground truth value
        """
        self.ground_truth = ground_truth

    @abstractmethod
    def get_data(self):
        """
        Should return the data associated with the underlying file.

        :return:
        """
        pass

    @abstractmethod
    def get_reader_info_dict(self) -> OrderedDict:
        """
        Should return an OrderedDict with information about the DataReader.

        :return: OrderedDict with information about the DataReader.
        """
        pass

    @abstractmethod
    def get_reader_info_str(self) -> str:
        """
        Should return a string with information about the DataReader.

        :return: String with information about the DataReader.
        """
        pass

    @abstractmethod
    def get_reader_identifier(self) -> str:
        """
        Should return a unique string identifier for the DataReader.

        :return: Unique string identifier for the DataReader.
        """
        pass

    def get_ground_truth(self):
        """
        :return: Returns the ground truth value.
        """
        return self.ground_truth


# Base class defining a PropertyExtractor for DataReader objects.

# Abstract methods:
#    .extract_property()
class ReaderPropertyExtractor(ABC):

    def __init__(self, property_name: str, *args, **kwargs):
        """
        :param property_name: Name of the property as a string.
        :param args         : Number of additional input arguments.
        :param kwargs       : Additional input arguments.
        """
        self.property_name = property_name
        super(ReaderPropertyExtractor, self).__init__(*args, **kwargs)

    @abstractmethod
    def extract_property(self, data_reader: DataReader):
        """
        Should extract the property using the DataReader as input.

        :param data_reader: DataReader object to extract the property from.
        :return           : Extracted property value.
        """
        pass

    @property
    def property_name(self):
        return self._property_name

    @property_name.setter
    def property_name(self, property_name):
        assert isinstance(property_name, str), \
            "{} Error: Property 'property_name' must be a string.".format(self.__class__.__name__)
        self._property_name = property_name


# Base class defining a MultiPropertyExtractor for DataReader objects.

# Available methods:
#   .__getitem__()
#   .__len__()
#   .has_property()
#   .get_property_names()
#   .extract_property()
class ReaderMultiPropertyExtractor(ABC):

    def __init__(self, property_extractors: (ReaderPropertyExtractor, list)):
        """
        :param property_extractors: Single ReaderPropertyExtractor or list of ReaderPropertyExtractors
        """
        self.property_extractors = property_extractors

    def __getitem__(self, property_name: str) -> ReaderPropertyExtractor:
        return self._get_property_extractor(property_name)

    def __len__(self):
        if self.property_extractors is not None:
            return len(self.property_extractors)
        return 0

    def has_property(self, property_name: str) -> bool:
        """
        Checks if a property exists.

        :param property_name: Property name as a string.
        :return             : Boolean variable specifying if property name is available.
        """
        self._validate_property_name_str(property_name)

        if property_name in self.property_extractors:
            return True
        return False

    def get_property_names(self) -> list:
        """
        :return: List of available property names.
        """
        return list(self.property_extractors.keys())

    def extract_property(self, data_reader: DataReader, property_name: str):
        """
        Extracts a property from a DataReader object.

        :param data_reader  : DataReader object to extract a property from.
        :param property_name: Property name to be extracted.
        :return             : The property value.
        """
        self._validate_property_name(property_name)

        return self.property_extractors[property_name].extract_property(data_reader)

    def _get_property_extractor(self, property_name: str) -> ReaderPropertyExtractor:

        self._validate_property_name(property_name)

        return self.property_extractors[property_name]

    def _validate_property_name(self, property_name: str):
        self._validate_property_name_str(property_name)

        assert self.has_property(property_name), \
            "{} Error: No property extractor exists with " \
            "'property_name'='{}'.". format(self.__class__.__name__, property_name)

    def _validate_property_name_str(self, property_name: str):
        assert isinstance(property_name, str), \
            "{} Error: Variable 'property_name' must be a string.".format(self.__class__.__name__)

    @property
    def property_extractors(self):
        return self._property_extractors

    @property_extractors.setter
    def property_extractors(self, property_extractors):
        if isinstance(property_extractors, ReaderPropertyExtractor):
            property_extractors = [property_extractors]

        assert isinstance(property_extractors, list) and all(
            list(map(lambda item: isinstance(item, ReaderPropertyExtractor), property_extractors))), \
            "{} Error: Property 'property_extractors' must be a list of 'ReaderPropertyExtractor' " \
            "objects.".format(self.__class__.__name__)

        assert len(property_extractors) > 0, \
            "{} Error: Property 'property_extractors' cannot be an empty list.".format(self.__class__.__name__)

        property_names = [property_extractor.property_name for property_extractor in property_extractors]
        assert len(set(property_names)) == len(property_names), \
            "{} Error: Duplicate property names found. All property_extractors must have a unique " \
            "'property_name'.".format(self.__class__.__name__)
        self._property_extractors = dict(zip(property_names, property_extractors))


# Base class defining a ReaderDataExtractor object for extracting data from a DataReader

# Abstract methods:
#    .get_property_extractors_validation_dictionary()
#    .extract_data()

# Available methods:
#    .has_property()
#    .has_property_extractors()
#    .extract_property()
class ReaderDataExtractor(ABC):

    def __init__(self, property_extractors: (None, list, ReaderPropertyExtractor,
                                             ReaderMultiPropertyExtractor) = None):
        """
        :param property_extractors: Available property extractors.
        """
        self.property_extractors = property_extractors

    @abstractmethod
    def get_property_extractors_validation_dictionary(self) -> (None, Callable, dict):
        pass

    @abstractmethod
    def extract_data(self, data_reader: DataReader):
        """
        Should extract data from the DataReader.

        :param data_reader: DataReader object
        :return           : Extracted data
        """
        pass

    def has_property(self, property_name: str) -> bool:
        """
        Checks if a property is available.

        :param property_name: Property name to check as a string.
        :return             : Boolean variable specifying if property exists.
        """
        if self.property_extractors is not None:
            return self.property_extractors.has_property(property_name)
        return False

    def has_property_extractors(self):
        """
        :return: True if ReaderDataExtractor has property extractors.
        """
        return self.property_extractors is not None

    def extract_property(self, data_reader: DataReader, property_name: str):
        """
        Extracts a property from the DataReader object.

        :param data_reader  : DataReader to extract the property from.
        :param property_name: Property name to be extracted as a string.
        :return             : Extracted property value.
        """
        assert self.has_property(property_name), \
            "{} Error: property '{}' does not exist.".format(self.__class__.__name__, property_name)

        return self.property_extractors.extract_property(data_reader, property_name)

    def _validate_property_extractors(self,
                                      property_extractors:
                                      (None, list, ReaderPropertyExtractor,
                                       ReaderMultiPropertyExtractor)) -> (None, ReaderMultiPropertyExtractor):

        if property_extractors is not None:
            if isinstance(property_extractors, (list, ReaderPropertyExtractor)):
                if isinstance(property_extractors, list) and len(property_extractors) == 0:
                    return None
                try:
                    property_extractors = ReaderMultiPropertyExtractor(property_extractors)
                except AssertionError:
                    print("{} Error: Property 'reader_property_extractors' is not "
                          "valid.".format(self.__class__.__name__ + "._validate_property_extractors()"))
                    raise

            assert isinstance(property_extractors, ReaderMultiPropertyExtractor), \
                "{} Error: Property 'reader_property_extractors' must be a 'ReaderMultiPropertyExtractor' " \
                "object.".format(self.__class__.__name__ + "._validate_property_extractors()")

            key_value_validation_dictionary = self.get_property_extractors_validation_dictionary()

            if key_value_validation_dictionary is None:
                return property_extractors

            assert isinstance(key_value_validation_dictionary, dict) or callable(key_value_validation_dictionary), \
                "{} Error: Method 'get_property_extractors_validation_dictionary()' must return a dictionary or " \
                "a callable object.".format(self.__class__.__name__ + "._validate_property_extractors()")

            if isinstance(key_value_validation_dictionary, dict):
                if len(key_value_validation_dictionary) == 0:
                    return property_extractors
                assert all(list(map(lambda item: item is None or callable(item),
                                    list(key_value_validation_dictionary.values())))), \
                    "{} Error: Dictionary returned by method " \
                    "'get_property_extractors_key_value_validation_dictionary()' contains invalid values. " \
                    "All values must be either None or callable " \
                    "objects.".format(self.__class__.__name__ + "._validate_property_extractors()")

                missing_property_names = [item for item in key_value_validation_dictionary.keys() if
                                          not property_extractors.has_property(item)]
                assert len(missing_property_names) == 0, \
                    "{} Error: Properties with names '{}' are missing from property " \
                    "'reader_property_extractors'.".format(self.__class__.__name__ + "._validate_property_extractors()",
                                                           ', '.join(["'" + item + "'"
                                                                      for item in missing_property_names]))

                property_names = property_extractors.get_property_names()
                invalid_property_names = [item for item in property_names if
                                          item not in key_value_validation_dictionary.keys()]
                assert len(invalid_property_names) == 0, \
                    "{} Error: Property 'reader_property_extractors' contains invalid property_names: [{}]. " \
                    "Valid property names are: " \
                    "[{}]".format(self.__class__.__name__ + "._validate_property_extractors()",
                                  ', '.join(["'" + item + "'" for item in invalid_property_names]),
                                  ', '.join(["'" + item + "'" for item in key_value_validation_dictionary.keys()]))
            else:
                validation_funcs = [key_value_validation_dictionary]*len(property_extractors)
                key_value_validation_dictionary = dict(zip(property_extractors.get_property_names(), validation_funcs))

            try:
                for key, validation_func in key_value_validation_dictionary.items():
                    if validation_func is not None:
                        validation_func(property_extractors[key])
            except Exception as e:
                raise RuntimeError("{} Error: Exception while validating property 'property_extractors'\n\t{}'.".
                                   format(self.__class__.__name__ + "._validate_property_extractors()", str(e)))

        return property_extractors

    @property
    def property_extractors(self):
        return self._property_extractors

    @property_extractors.setter
    def property_extractors(self, property_extractors):
        property_extractors = self._validate_property_extractors(property_extractors)
        self._property_extractors = property_extractors


# Base class for defining a Dataset. Can construct a list of DataReaders which can then be
# used to extract data.

# Abstract methods:
#    .create_reader()
#    .load_partitions()

# Available methods:
#    .get_partition()
class Dataset(ABC):
    def __init__(self, db_path, ground_truth_path, dataset_partition_path):
        """
        :param db_path               : Path to database where data is stored.
        :param ground_truth_path     : Path to ground truth.
        :param dataset_partition_path: Path to partition.
        """
        super(Dataset, self).__init__()
        self.db_path = db_path
        self.ground_truth_path = ground_truth_path
        self.dataset_partition_path = dataset_partition_path
        self.partitions = None

    @abstractmethod
    def create_reader(self, reader_entries: dict) -> DataReader:
        """
        Should construct a DataReader using a reader_entries dictionary.

        :param reader_entries: Dictionary with information about constructing the DataReader.
        :return              : DataReader object.
        """
        pass

    # @abstractmethod
    # def load_ground_truth(self):
    #     pass

    @abstractmethod
    def load_partitions(self) -> dict:
        pass

    def _get_data_partition(self, set_name: str="train") -> list:
        if not self.partitions:
            self.partitions = self.load_partitions()
        if set_name in self.partitions:
            partition = self.partitions[set_name]

            return self._create_data_readers(partition)
        else:
            return []

    def get_partition(self, set_name: str) -> list:
        return self._get_data_partition(set_name)

    def _create_data_readers(self, partition: list) -> list:
        data_readers = []

        assert isinstance(partition, list) and all(list(map(lambda item: isinstance(item, dict), partition))), \
            "{}.create_data_readers() Error: Variable 'partition' must be a list of " \
            "dictionaries.".format(self.__class__.__name__ + "._create_data_readers()")

        for row in partition:
            try:
                reader = self.create_reader(row)
                data_readers.append(reader)
            except Exception as e:
                print("{}.create_data_readers() Error: {}. Reader creation "
                      "skipped.".format(self.__class__.__name__ + "._create_data_readers()", str(e)))

        assert all(list(map(lambda item: isinstance(item, DataReader), data_readers))), \
            "{}.create_data_readers() Error: Variable 'data_readers' must be a list of 'DataReader' " \
            "instances.".format(self.__class__.__name__ + "._create_data_readers()")

        return data_readers

    @property
    def partitions(self) -> (None, dict):
        return self._partitions

    @partitions.setter
    def partitions(self, partitions: (None, dict)):
        if partitions is not None:
            assert isinstance(partitions, dict), "Method 'load_partitions' must return a dictionary."
        self._partitions = partitions


# Base class extending the base class Dataset to a CsvBiometricDataset

# Abstract methods:
#    .create_reader()
#    .is_pa()

# Available methods:
#    .get_partition()
#    .load_partitions()
class CsvBiometricDataset(Dataset):
    VALID_MODALITIES = ['FINGER', 'FACE', 'IRIS']
    VALID_PARTITION_NAMES = ['TRAIN', 'TEST', 'VALID']

    def __init__(self, db_path: str, ground_truth_path: str,
                 dataset_partition_path: str, modality: str,
                 needed_ground_truth_entries: (None, tuple) = None,
                 optional_ground_truth_entries: (None, tuple) = None,
                 strict_keys: bool = True):
        """
        :param db_path                      : Path to directory where data is stored.
        :param ground_truth_path            : Path to a .csv file containing ground truth information.
        :param dataset_partition_path       : Path to a .csv file defining partitions.
        :param modality                     : Biometric modality (one of ['FACE', 'FINGER', 'IRIS'].
        :param needed_ground_truth_entries  : Tuple of strings specifying required fields in the provided
                                              ground truth .csv file.
                                  (Optional)  Default is None.
        :param optional_ground_truth_entries: Tuple of string specifying optional fields in the provided ground truth
                                              .csv file.
                                  (Optional)  Default is None.
        :param strict_keys                  : Boolean flag specifying if keys are strict. If strict, all
                                              fields in the partition .csv file except for 'partition_name' are assumed
                                              as required and augment the 'needed_ground_truth_entries'. As such, the
                                              partition .csv must contain fields that exist in the ground truth plus an
                                              additional 'partition_name' entry. If False, this requirement is lifted,
                                              the partition .csv can contain additional entries which will be passed
                                              to the reader - instead of being found in the ground_truth .csv. In such
                                              a way, the ground_truth .csv can contain a subset of fields of the
                                              partition_csv and only the common fields between the partition .csv and
                                              the ground_truth .csv will be used as keys. The remaining data from the
                                              non common fields will be added to the reader from the partition .csv.
                                  (Optional)  Default is True.

        It is assumed that the partition .csv file should contain a column with name 'partition_name'.
        The common fields in the provided ground truth and partition .csv files will be used as keys to index each
        partition entry to its corresponding ground truth entry.
        The needed_ground_truth_entries and optional_ground_truth_entries (if found) will be used to construct a
        dictionary which will be passed to the .create_reader() method which should construct the DataReader object.
        """
        super(CsvBiometricDataset, self).__init__(db_path, ground_truth_path, dataset_partition_path)
        self.modality = modality
        self.needed_ground_truth_entries = needed_ground_truth_entries
        self.optional_ground_truth_entries = optional_ground_truth_entries
        self.strict_keys = strict_keys

    @abstractmethod
    def create_reader(self, reader_entries: dict) -> DataReader:
        pass

    @abstractmethod
    def is_pa(self, pai, modality: (None, str) = None):
        pass

    @staticmethod
    def create_gt_dictionary_key(gt_dict: dict, keys: list) -> str:
        return '-'.join(list(map(lambda key: gt_dict[key], keys)))

    def load_partitions(self) -> dict:
        """
        :return: Each partition (list of DataReaders) as dictionary with keys each partition name.
        """
        partitions = {}
        with open(self.dataset_partition_path, "r") as csv_file:
            reader = csv.DictReader(csv_file)
            assert 'partition_name' in reader.fieldnames, \
                "{} Error: csv_file: {} must contain a 'partition_name' " \
                "column.".format(self.__class__.__name__ + ".load_partitions()", self.dataset_partition_path)
            keys = [key for key in reader.fieldnames if key != 'partition_name']
            needed_gt_entries = self.needed_ground_truth_entries
            if self.strict_keys:
                for key in keys:
                    if key not in needed_gt_entries:
                        needed_gt_entries.append(key)
                reader_keys = None
            else:
                reader_keys = [key for key in keys if key not in needed_gt_entries]
                if len(reader_keys) == 0:
                    reader_keys = None
                keys = [key for key in keys if key in needed_gt_entries]
                assert len(keys) > 0, \
                    "{} Error: Operating with 'strict_keys' = False. No common keys found between partition and " \
                    "ground_truth .csv files.".format(self.__class__.__name__ + ".load_partitions()")
            self.needed_ground_truth_entries = tuple(needed_gt_entries)
            ground_truth = self._create_ground_truth_dictionary(keys)
            for r in reader:
                partition_name = r["partition_name"]
                assert partition_name.upper() in self.VALID_PARTITION_NAMES or partition_name.upper().startswith(
                    'FOLD'), \
                    "{} RuntimeError: 'partition_name'={} is invalid. Valid partition names are: {} " \
                    "or a name starting with 'FOLD' as " \
                    "'FOLD*'.".format(self.__class__.__name__ + ".load_partitions()",
                                      partition_name,
                                      ', '.join(["'" + item + "'" for
                                                 item in CsvBiometricDataset.VALID_PARTITION_NAMES]))
                if partition_name not in partitions:
                    partitions[partition_name] = []

                p = dict()
                # Create ground truth dictionary key
                gt_key = CsvBiometricDataset.create_gt_dictionary_key(r, keys)

                # Create needed dictionary items
                for key in self.needed_ground_truth_entries:
                    p[key] = ground_truth[gt_key][key]

                # Create optional dictionary items
                if self.optional_ground_truth_entries is not None:
                    for key in self.optional_ground_truth_entries:
                        if key not in p:
                            if key in ground_truth[gt_key]:
                                p[key] = ground_truth[gt_key][key]
                            else:
                                p[key] = None

                if reader_keys is not None:
                    for key in reader_keys:
                        p[key] = r[key]

                partitions[partition_name].append(p)

        for key, val in partitions.items():
            print("\tThe size of the {} set is {}".format(key, len(val)))
        return partitions

    def _load_ground_truth(self) -> list:
        """
        Reads the ground truth .csv file and returns a list whose elements are dictionaries created by the entries
        of each row of the .csv file.

        :return: List of dictionaries.
        """
        ground_truth = []
        with open(self.ground_truth_path, "r") as csv_file:
            reader = csv.DictReader(csv_file)
            if self.needed_ground_truth_entries is not None:
                assert all([item in reader.fieldnames for item in self.needed_ground_truth_entries]), \
                    "{} Error: Ground truth file {} must contain entries {}.".format(
                        self.__class__.__name__ + "._load_ground_truth()",
                        self.ground_truth_path,
                        ', '.join(["'" + item + "'" for item in self.needed_ground_truth_entries]))
            for r in reader:
                ground_truth.append(dict(r))

        return ground_truth

    def _create_ground_truth_dictionary(self, keys: list) -> dict:
        """
        Converts the list of dictionaries extracted from the ground truth .csv file to a dictionary whose
        keys are created the fieldnames that are common between the ground truth and partition .csv files.

        :param keys: List of string specifying the fields of the ground truth .csv file to be used as keys.
        :return    : Ground truth dictionary.
        """
        ground_truth = dict()
        for gt in self._load_ground_truth():
            gt_key = CsvBiometricDataset.create_gt_dictionary_key(gt, keys)
            assert gt_key not in ground_truth, \
                "{} Error: Duplicate ground truth key '{}' " \
                "detected.".format(self.__class__.__name__ + "._create_ground_truth_dictionary()", gt_key)
            ground_truth[gt_key] = gt

        return ground_truth

    @property
    def db_path(self) -> str:
        return self._db_path

    @db_path.setter
    def db_path(self, db_path: str) -> None:
        if db_path is not None:
            assert isinstance(db_path, str), \
                "{} Error: Property 'db_path' must be a string.".format(self.__class__.__name__)
            assert os.path.isdir(db_path), "{} Error: Directory 'db_path'={} does not " \
                                           "exist.".format(self.__class__.__name__, db_path)
        self._db_path = db_path

    @property
    def ground_truth_path(self) -> str:
        return self._ground_truth_path

    @ground_truth_path.setter
    def ground_truth_path(self, ground_truth_path: str) -> None:

        assert isinstance(ground_truth_path, str), \
            "{} Error: Property 'ground_truth_path' must be a string.".format(self.__class__.__name__)
        assert os.path.isfile(ground_truth_path), \
            "{} Error: File 'ground_truth_path'={} does not " \
            "exist.".format(self.__class__.__name__, ground_truth_path)
        _, ext = os.path.splitext(ground_truth_path)
        assert ext == '.csv', "{} Error: File 'ground_truth_path'={} must be a '.csv' " \
                              "file.".format(self.__class__.__name__, ground_truth_path)
        self._ground_truth_path = ground_truth_path

    @property
    def dataset_partition_path(self) -> (None, str):
        return self._dataset_partition_path

    @dataset_partition_path.setter
    def dataset_partition_path(self, dataset_partition_path: (None, str)) -> None:
        assert isinstance(dataset_partition_path, str), \
            "{} Error: Property 'dataset_partition_path' must be a " \
            "string.".format(self.__class__.__name__)
        assert os.path.isfile(dataset_partition_path), \
            "{} Error: File 'dataset_partition_path'={} does not " \
            "exist.".format(self.__class__.__name__, dataset_partition_path)
        _, ext = os.path.splitext(dataset_partition_path)
        assert ext == '.csv', "{} Error: File 'dataset_partition_path'={} must be a '.csv' " \
                              "file.".format(self.__class__.__name__, dataset_partition_path)
        self._dataset_partition_path = dataset_partition_path

    @property
    def modality(self) -> str:
        return self._modality

    @modality.setter
    def modality(self, modality: str) -> None:
        assert isinstance(modality, str), "Property 'modality' must be a string."
        assert modality.upper() in CsvBiometricDataset.VALID_MODALITIES, \
            "{} Error: Property 'modality'={} must contain a valid modality. Valid modalities are: " \
            "{}".format(self.__class__.__name__, modality,
                        ', '.join(["'" + item + "'" for item in CsvBiometricDataset.VALID_MODALITIES]))
        self._modality = modality

    @property
    def needed_ground_truth_entries(self) -> list:
        return list(self._needed_ground_truth_entries)

    @needed_ground_truth_entries.setter
    def needed_ground_truth_entries(self, needed_ground_truth_entries: (None, tuple)) -> None:
        if needed_ground_truth_entries is not None:
            assert isinstance(needed_ground_truth_entries, tuple), \
                "{} Error: Property 'needed_ground_truth_entries' must be a tuple.".format(self.__class__.__name__)
            assert all(tuple(map(lambda item: isinstance(item, str) and len(item) > 0, needed_ground_truth_entries))), \
                "{} Error: Property 'needed_ground_truth_entries' must be a list of nonempty " \
                "strings.".format(self.__class__.__name__)
            assert len(set(needed_ground_truth_entries)) == len(needed_ground_truth_entries), \
                "{} Error: Property 'needed_ground_truth_entries' must contain unique " \
                "entries.".format(self.__class__.__name__)
        else:
            needed_ground_truth_entries = []
        self._needed_ground_truth_entries = list(needed_ground_truth_entries)

    @property
    def optional_ground_truth_entries(self) -> (None, tuple):
        return self._optional_ground_truth_entries

    @optional_ground_truth_entries.setter
    def optional_ground_truth_entries(self, optional_ground_truth_entries: (None, tuple)) -> None:
        if optional_ground_truth_entries is not None:
            assert isinstance(optional_ground_truth_entries, tuple), \
                "{} Error: Property 'optional_ground_truth_entries' must be a tuple.".format(self.__class__.__name__)
            assert all(tuple(map(lambda item: isinstance(item, str) and len(item) > 0,
                                 optional_ground_truth_entries))), \
                "{} Error: Property 'optional_ground_truth_entries' must be a tuple of nonempty " \
                "strings.".format(self.__class__.__name__)
            assert len(set(optional_ground_truth_entries)) == len(optional_ground_truth_entries), \
                "{} Error: Property 'optional_ground_truth_entries' must contain unique " \
                "entries".format(self.__class__.__name__)
        self._optional_ground_truth_entries = optional_ground_truth_entries

    @property
    def strict_keys(self) -> bool:
        return self._strict_keys

    @strict_keys.setter
    def strict_keys(self, strict_keys: bool) -> None:
        assert isinstance(strict_keys, bool), \
            "{} Error: Property 'strict_keys' must be boolean.".format(self.__class__.__name__)
        self._strict_keys = strict_keys


class BiometricMultiLabelData(ABC):
    def __init__(self, pai_code_label_mapping: (None, dict) = None):
        self.pai_code_label_mapping = pai_code_label_mapping
        if self.pai_code_label_mapping is None:
            self.pai_code_label_mapping = self.get_default_pai_code_label_mapping()
        self.pai_key_len = self._get_pai_key_len()

    @abstractmethod
    def get_default_pai_code_label_mapping(self) -> dict:
        pass

    def _get_pai_key_len(self) -> (None, int):
        key_len = list(map(len, self.pai_code_label_mapping.keys()))
        if all([key_len[i+1] == key_len[i] for i in range(len(key_len)-1)]):
            return key_len[0]
        return None

    def get_num_classes(self) -> int:
        return len(list(set(self.pai_code_label_mapping.values())))

    def get_pai_label(self, pai: str) -> int:
        assert isinstance(pai, str), \
            "{}.get_pai_label() Error: Variable 'pai' must be a " \
            "string.".format(self.__class__.__name__)
        if self.pai_key_len is not None:
            assert len(pai) >= self.pai_key_len, \
                "{}.get_pai_label() Error: Variable 'pai' must have at least {} number of " \
                "characters.".format(self.__class__.__name__, self.pai_key_len)
            try:
                label_val = self.pai_code_label_mapping[pai[:self.pai_key_len]]
            except BaseException:
                raise RuntimeError("Provided pai code {} does match any entries "
                                   "in the 'pai_code_label_mapping' dictionary.".format(pai))
        else:
            label_val = None
            for key, val in self.pai_code_label_mapping.items():
                if pai.startswith(key):
                    label_val = val
                    break
            if label_val is None:
                raise RuntimeError("Provided pai code {} does not match any "
                                   "entries in the 'pai_code_label_mapping' dictionary.".format(pai))

        return label_val

    @property
    def pai_code_label_mapping(self) -> (None, dict):
        return self._pai_code_label_mapping

    @pai_code_label_mapping.setter
    def pai_code_label_mapping(self, pai_code_label_mapping: dict) -> None:
        if pai_code_label_mapping is not None:
            assert isinstance(pai_code_label_mapping, dict), \
                "{} Error: Property 'pai_code_label_mapping' must be a dictionary.".format(self.__class__.__name__)
            vals = list(pai_code_label_mapping.values())
            assert all(list(map(lambda item: isinstance(item, int), vals))), \
                "{} Error: Property 'pai_code_label_mapping' must be a dictionary with integer " \
                "values.".format(self.__class__.__name__)
            vals = list(set(vals))
            vals.sort()
            assert min(vals) == 0 and all(
                [item == 1 for item in [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]]), \
                "{} Error: Property 'pai_code_label_mapping' must be a dictionary with consecutive integer " \
                "values between 0 and {}.".format(self.__class__.__name__, len(vals) - 1)
        self._pai_code_label_mapping = pai_code_label_mapping

    @property
    def pai_key_len(self) -> (None, int):
        return self._pai_key_len

    @pai_key_len.setter
    def pai_key_len(self, pai_key_len: (None, int)) -> None:
        if pai_key_len is not None:
            assert pai_key_len > 0, "{} Error: Property 'pai_key_len' must be a positive " \
                                    "integer.".format(self.__class__.__name__)
        self._pai_key_len = pai_key_len


# Extends the base class Dataset to a MultiDataset that extends multiple Datasets into one.
class MultiDataset(Dataset):
    def __init__(self, datasets: list):
        """
        :param datasets: List of Dataset objects to be merged.
        """
        super(MultiDataset, self).__init__(None, None, None)
        self.datasets = datasets

    def get_partition(self, partition_name: str) -> list:
        """
        Combines the partition from each dataset into a single list of DataReader objects.

        :param partition_name: Name of partition to extract.
        :return              : List of DataReader objects for the specified partition.
        """
        partition = []
        for i, ds in enumerate(self.datasets):
            print("Dataset {}:".format(i))
            partition.extend(ds.get_partition(partition_name))

        print("\nThe total size of the {} set is {}\n".format(partition_name,
                                                              len(partition)))

        return partition

    def create_reader(self, reader_entries: dict):
        raise RuntimeError("Not implemented yet")

    def load_partitions(self):
        raise RuntimeError("Not implemented yet")

    @property
    def datasets(self) -> list:
        return self._datasets

    @datasets.setter
    def datasets(self, datasets: list) -> None:
        assert isinstance(datasets, list) and all(
            list(map(lambda item: isinstance(item, Dataset), datasets))), \
            "{} Error: Input 'datasets' must be a list of 'Dataset' instances.".format(
                self.__class__.__name__)
        self._datasets = datasets
