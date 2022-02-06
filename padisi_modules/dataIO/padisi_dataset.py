import os
from abc import abstractmethod
from collections import OrderedDict
from typing import Callable

import numpy as np
import tables
from scipy.spatial.distance import cdist

from padisi_modules.dataIO.dataset import DataReader, CsvBiometricDataset, MultiDataset, BiometricMultiLabelData, \
    ReaderDataExtractor, ReaderPropertyExtractor, ReaderMultiPropertyExtractor


# Helper functions

def is_padisi_gt_pa(pai: str, modality: str) -> bool:
    """
    Function that returns True/False if a specific code corresponds to a PAI for a biometric modality.

    :param pai     : Ground truth code as a string.
    :param modality: Biometric modality. One of ['FACE', 'FINGER', 'IRIS'].
    :return        : True if code corresponds to a PAI, else False.
    """
    assert isinstance(pai, str) and len(pai) > 0, "Variable 'pai' must be a nonempty string"
    assert isinstance(modality, str) and len(modality) > 0, "Variable 'modality' must be a nonempty string."

    pai_to_check = pai.lower()
    modality = modality.upper()

    if pai_to_check.startswith("m5") and modality == "FACE":
        return True
    if pai_to_check.startswith("m4") and modality == "FINGER":
        return True
    if pai_to_check.startswith("m6") and modality == "IRIS":
        return True
    if pai_to_check.startswith("m0"):
        return False
    if not pai_to_check.startswith("m4") and \
            not pai_to_check.startswith("m5") and \
            not pai_to_check.startswith("m6") and \
            not pai_to_check.startswith("m0"):
        raise RuntimeError("Invalid PAI code: '{}' for modality: '{}'".format(pai, modality))
    return False


FINGER_NAME_TO_POSITION_DICT = {'THUMB': 1, 'INDEX': 2, 'MIDDLE': 3, 'RING': 4, 'LITTLE': 5}


def increment_trial_id_last_char(trial_id: str) -> str:
    """
    Helper function to modify the trial id by incrementing its last character by one.
    Allows only alphanumeric lowercase characters. Used for modifying the RIGHT_IRIS trial_id and assign
    a unique new trial_id for RIGHT_IRIS.

    :param trial_id: Trial ID as a string.
    :return        : Modified trial id with its last character incremented by one.
    """
    assert isinstance(trial_id, str) and len(trial_id) > 0, \
        "Variable 'trial_id' must be a non-empty string."

    last_char = trial_id[-1]
    if last_char == 'z':
        last_char = '0'
    else:
        last_char = chr(ord(last_char) + 1)
        while not (last_char.isalnum() and (last_char.islower() if last_char.isalpha() else True)):
            last_char = chr(ord(last_char) + 1)

    return trial_id[:-1] + last_char


def decrement_trial_id_last_char(trial_id: str) -> str:
    """
    Helper function to modify the trial id by decrementing its last character by one.
    Allows only alphanumeric lowercase characters. Used for reverting the effect of function
    'increment_trial_id_last_char' above.

    :param trial_id: Trial ID as a string.
    :return        : Modified trial id with its last character decremented by one.
    """
    assert isinstance(trial_id, str) and len(trial_id) > 0, \
        "Variable 'trial_id' must be a non-empty string."

    last_char = trial_id[-1]
    if last_char == '0':
        last_char = 'z'
    else:
        last_char = chr(ord(last_char) - 1)
        while not (last_char.isalnum() and (last_char.islower() if last_char.isalpha() else True)):
            last_char = chr(ord(last_char) - 1)

    return trial_id[:-1] + last_char


HARDWARE_SYNCED_DATASETS_STRS = ('BASLER_LEFT_NIR', 'BASLER_RIGHT_NIR', 'BASLER_RGB', 'BASLER_BGR', 'BOBCAT_SWIR')


def is_dataset_hardware_synced(collection_id: (None, str), dataset_name: str) -> tuple:
    """
    Checks if a particular dataset is hardware synced. Only useful for FACE data.

    :param collection_id: Collection id string of the data reader.
    :param dataset_name : Dataset name string.
    :return             : Tuple with (start string of dataset name, True if dataset is hardware synced).
    """
    if collection_id is not None:
        assert isinstance(collection_id, str) and len(collection_id) > 0, \
            "{} Error: Variable 'collection_id' must be a non-empty string.".format('is_dataset_hardware_synced()')

    assert isinstance(dataset_name, str) and len(dataset_name) > 0, \
        "{} Error: Variable 'dataset_name' must be a non-empty string.".format('is_dataset_hardware_synced()')

    flags = [dataset_name.upper().startswith(item) for item in HARDWARE_SYNCED_DATASETS_STRS]
    if any(flags):
        return HARDWARE_SYNCED_DATASETS_STRS[flags.index(True)], True

    return None, False


def open_h5_file_for_reading(path: str) -> tables.file.File:
    try:
        h5_file = tables.open_file(path, mode='r')
        return h5_file
    except OSError:
        raise


def close_h5_file(h5_file: (None, tables.file.File)) -> None:
    if h5_file is not None:
        if h5_file.isopen:
            h5_file.close()


# Extension of base class DataReader for PADISI H5 data. Used to extract data from .h5 files of the PADISI dataset.
#
# Available methods:
#    .open_reader()
#    .close_reader()
#    .__enter__()
#    .__exit__()
#    .get_reader_info_dict()
#    .get_reader_info_str()
#    .get_reader_identifier()
#    .get_list_of_trial_ids()
#    .get_data()
#    .get_available_dataset_names()
#    .get_dataset_num_frames()
#    .get_dataset_frames()
#    .get_synced_frames()
#    .get_available_dataset_attribute_names()
#    .get_dataset_attribute()
#    .get_dataset_wavelength()
#    .get_dataset_timestamps()
#    .get_collection_id()
class PadisiH5Reader(DataReader):

    DATE_ATTR_STRS = ('datetime', 'date_time')
    DATE_COLLECTION_ID_MAPPING = {'2019-02': 'USC1', '2019-08-15': 'USC2'}
    REQUIRED_CSV_READER_ENTRIES = ['transaction_id', 'trial_id', 'modality', 'ground_truth', 'trial_name',
                                   'relative_path']
    TRANSACTION_LEVEL_GROUP_NAMES = ['finger', 'face', 'iris', 'sensor']
    VALID_PAI_START_CHARS = ['m0', 'm4', 'm5', 'm6']

    def __init__(self, db_path: (None, str), transaction_id: str, trial_id: str, pai: str,
                 ground_truth, modality: str, trial_name: (None, str),
                 data_extractor: (None, ReaderDataExtractor, tuple, list) = None, check_existence: bool = False):
        """
        :param db_path        : Full path to the .h5 file containing the data for this reader.
        :param transaction_id : Transaction id string.
        :param trial_id       : Trial id string.
        :param pai            : Ground truth code string.
        :param ground_truth   : Ground truth label (e.g., boolean, integer).
        :param modality       : Modality string (e.g., 'finger').
        :param trial_name     : Trial name string or None.
        :param data_extractor : ReaderDataExtractor object to be used when the .get_data() method is invoked.
                    (Optional)  Default is None.
        :param check_existence: Boolean variable specifying whether the underlying .h5 file actually exists.
                                Can be set to False if data has been extracted but one needs access to other properties
                                within the reader object. Note that if check_existence = False, the collection_id will
                                be None and not available to the PadisiH5Reader.
                    (Optional)  Default is True.
        """
        super(PadisiH5Reader, self).__init__(ground_truth)
        self.h5_file = None
        self.db_path = db_path
        self.transaction_id = transaction_id
        self.trial_id = trial_id
        self.pai = pai
        self.modality = modality
        self.trial_name = trial_name
        self.data_extractor = data_extractor

        assert isinstance(check_existence, bool), \
            "{} Error: Property 'check_existence' must be boolean.".format(self.__class__.__name__)

        if check_existence:
            self._verify_existence()
            self.collection_id = self.get_collection_id()

    def open_reader(self):
        """
        :return: The PadisiH5Reader object itself.
                 Allows for use of reader within a with statement as:
                    with reader.open_reader() as h5_reader:
                        --- operations with h5_reader ---
        """
        return self

    def __enter__(self):
        """
        :return: The PadisiH5Reader object itself.
                 Allows for use of reader within a with statement as:
                    with reader.open_reader() as h5_reader:
                         --- operations with h5_reader ---
        """
        return self.open_reader()

    def __exit__(self, *exc_info):
        """
        Closes the h5_file within the reader.

        :param exc_info: execution info, passed to the __exit__ magic method by default.
        :return:
                 Allows for use of reader within a with statement as:
                    with reader.open_reader() as h5_reader:
                         --- operations with h5_reader ---
                 Ensures that the h5_file of the reader is closed at the end of the with statement.
        """
        self.close_reader()
        return False

    def get_reader_info_dict(self) -> OrderedDict:
        """
        Returns information about the reader as a dictionary.

        :return: An OrderedDict with the main information of the reader:
                 - transaction_id.
                 - trial_id.
                 - trial_name (if not None).
                 - modality (if not None).
                 - ground_truth (the pai code).
        """
        reader_info_dict = OrderedDict()
        reader_info_dict['transaction_id'] = self.transaction_id
        reader_info_dict['trial_id'] = self.trial_id
        if self.trial_name is not None:
            reader_info_dict['trial_name'] = self.trial_name
        if self.modality is not None:
            reader_info_dict['modality'] = self.modality
        reader_info_dict['ground_truth'] = self.pai

        return reader_info_dict

    def get_reader_info_str(self) -> str:
        """
        Returns information about the reader as a string.

        :return: A string with information about the reader. Useful when running a loop on readers and
                 the user wants to print information about the current reader.
        """
        info_str = "Tx Id: {}, Tr Id: {}".format(self.transaction_id, self.trial_id)
        if self.trial_name is not None:
            info_str += ", Tr Name: {}".format(self.trial_name)

        info_str += ", PAI Code: {}".format(self.pai)
        return info_str

    def get_reader_identifier(self) -> str:
        """
        Returns a unique string identifier of the reader.

        :return: A string uniquely identifying the reader. Some modifications are performed for this ID to be consistent
                 with python's dictionary key naming rules.
        """
        identifier = "_".join(['ID', self.transaction_id, self.trial_id])
        if self.trial_name is not None:
            identifier += "_" + self.trial_name

        return identifier

    def get_list_of_trial_ids(self) -> list:
        """
        Returns available trial_ids within the h5_file.

        :return: List of trial_ids in the current h5 file, or a single trial_id if the file is trial level.
        """
        path = self._get_file_path()
        h5_file = None
        try:
            h5_file = open_h5_file_for_reading(path)
            groups_names = list(getattr(h5_file.get_node(where='/'), '_v_groups').keys())
            trial_ids = [item for item in groups_names if item.lower() not in
                         PadisiH5Reader.TRANSACTION_LEVEL_GROUP_NAMES]
        except Exception as e:
            print("{}: Exception during extracting list of trial_ids of hdf5 file.: \n\t{}".
                  format(self.__class__.__name__ + ".get_list_of_trial_ids()", str(e)))
            trial_ids = []
        finally:
            close_h5_file(h5_file)

        return trial_ids

    def get_data(self) -> (np.ndarray, list):
        """
        General method that returns data from the h5_file in different formats based on the reader properties.

        :return:
            - If 'data_extractor' is a ReaderDataExtractor object: Returns the np.ndarray of the
              ReaderDataExtractor.extract_data() method.
            - If 'data_extractor' is a tuple or list of ReaderDataExtractor objects: Returns a list of the
              np.ndarrays that each ReaderDataExtractor.extract_data() method in the sequence returns.
        """
        assert self.data_extractor is not None, \
            "{} Error: Method .get_data() cannot be used if property 'data_extractor' is " \
            "None.".format(self.__class__.__name__)

        if isinstance(self.data_extractor, (tuple, list)):
            data = [data_extractor.extract_data(self) for data_extractor in self.data_extractor]
        else:
            data = self.data_extractor.extract_data(self)

        return data

    def get_dataset_num_frames(self, dataset_name: str) -> int:
        """
        Returns the total number frames for a dataset in the h5_file.

        :param dataset_name: Dataset name of interest (e.g., "BASLER_RGB_BGR")
        :return            : Total number of frames for the specified dataset in 'dataset_name'.
                             If dataset does not exist in the .h5 file, it returns 0.
        """
        self._validate_dataset_name_str(dataset_name)

        path = self._get_file_path()
        h5_file = None
        try:
            dataset_names = self._assert_dataset_available(dataset_name)

            h5_file = open_h5_file_for_reading(path)

            full_trial_id = self._get_full_trial_id(h5_file, self.trial_id)
            node = h5_file.get_node(where='/', name=full_trial_id)
            dataset_path = list(getattr(node, '_v_links').values())[dataset_names.index(dataset_name)]
            dataset_node = h5_file.get_node(where=dataset_path)

            num_frames = dataset_node.shape[0]
        except Exception as e:
            print("{}: Exception during extracting frames from dataset {} of hdf5 file.: \n\t{}".
                  format(self.__class__.__name__ + ".get_dataset_frames()", dataset_name, str(e)))
            num_frames = 0
        finally:
            close_h5_file(h5_file)

        return num_frames

    def get_dataset_frames(self, dataset_name: str, frames: (None, list) = None) -> np.ndarray:
        """
        Returns a set of frames from a dataset in the h5_file.

        :param dataset_name: Dataset name of interest (e.g., "BASLER_RGB_BGR")
        :param frames      : Frames to read from the dataset as a list of integers (e.g., [0, 4, 10]).
                             If None, all frames are read.
                 (Optional)  Default is None.
        :return            : A np.ndarray containing all frames specified by indices in 'frames' for dataset specified
                             by 'dataset_name'. It is a 4D array with dimensions: (num_frames, channels, height, width).
                             If the dataset specified by 'dataset_name' does not exist of the requested frames by
                             'frames' are out of bounds, the method raises an Exception. Use the
                             .get_dataset_num_frames() method to identify how many frames the dataset contains or 0
                             if the dataset does not exist.
        """
        self._validate_dataset_name_str(dataset_name)

        if frames is not None:
            assert isinstance(frames, list) and all(
                list(map(lambda item: (isinstance(item, int) or isinstance(item, np.integer)) and item >= 0,
                         frames))), \
                "{} Error: Variable 'frames' must be a non-empty list of non-negative " \
                "integers.".format(self.__class__.__name__ + ".get_dataset_frames()")

            if len(frames) == 0:
                frames = None

        dataset_names = self._assert_dataset_available(dataset_name)

        path = self._get_file_path()
        h5_file = None
        try:
            h5_file = open_h5_file_for_reading(path)
            full_trial_id = self._get_full_trial_id(h5_file, self.trial_id)
            node = h5_file.get_node(where='/', name=full_trial_id)
            dataset_path = list(getattr(node, '_v_links').values())[dataset_names.index(dataset_name)].target
            dataset_node = h5_file.get_node(where=dataset_path)

            if frames is not None:
                num_frames = dataset_node.shape[0]
                max_frame = max(frames)
                assert (max_frame < num_frames), \
                    "{} Error: Requested frames are out of bounds. Total number of frames for dataset '{}' is: " \
                    "{}.".format(self.__class__.__name__ + ".get_dataset_frames()", dataset_name, num_frames)
                frames = [dataset_node.read(start=item, stop=item+1) for item in frames]
                frames = np.concatenate(frames, axis=0)
            else:
                frames = dataset_node.read()
        except Exception as e:
            raise RuntimeError("{} Exception during extracting frames from dataset {} of hdf5 file.: \n\t{}".
                               format(self.__class__.__name__ + ".get_dataset_frames()", dataset_name, str(e)))
        finally:
            close_h5_file(h5_file)

        return frames

    def get_synced_frames(self, reference_dataset: str, target_datasets: (str, list),
                          modify_hardware_synced_timestamps: bool = False) -> tuple:
        """
        Returns frame indices that are synchronized (captured at the same or closest time) for a set of datasets
        in the h5_file. Synchronization uses a dataset as a reference and synchronized frames are found for all
        remaining datasets of interest.

        :param reference_dataset                : Reference dataset of interest (e.g., "BASLER_RGB_BGR")
        :param target_datasets                  : Target dataset(s) of interest. Can be one (string) or more
                                                  (list of strings).
        :param modify_hardware_synced_timestamps: Boolean variable specifying if timestamps of hardware synced datasets
                                                  should be modified. The timestamps are not 100% accurate since they
                                                  capture the time a frame finished processing from a camera and depends
                                                  on how fast the camera processes the frame. If this value is set to
                                                  True, timestamps for hardware synced datasets are modified to start
                                                  with a common timestamp (since it is known that these cameras started
                                                  capturing at the same time) removing the inaccuracy resulting from
                                                  the software calculation of the timestamps. Unfortunately, this is
                                                  not possible for software triggered cameras. This is mainly useful
                                                  for FACE data.
                                       (Optional) Default is False.
        :return                 : Returns a tuple with two values:
                                     - 'full_overlap_found': Boolean flag specifying if overlap was found between the
                                                             reference and target datasets. If False, it means that
                                                             frames were captured asynchronously and the returned
                                                             synced frames are the closest ones in time, but not
                                                             necessarily reliable.
                                     - 'frames_idx'        : list of lists containing the indices of the frames that
                                                             are synchronized from each dataset as:
                                                             [[reference_dataset_frame_indices],
                                                              [target_dataset_1_frame_indices] , ...
                                                              [target_dataset_n_frame_indices]].
                                  e.g., an output of [[0, 4], [1, 2], [4, 5]] means:
                                  Frame 0 of the reference_dataset is closest in time to frame 1 of target_dataset_1
                                  and frame 4 of target_dataset_2.
                                  Frame 4 of the reference dataset is closest in time to frame 2 of target_dataset_1
                                  and frame 5 of target_dataset_2.
        """
        try:
            self._validate_dataset_name_str(reference_dataset)
            assert isinstance(target_datasets, str) or isinstance(target_datasets, list), \
                "Error: Variable 'target_datasets' must be a string or a list."
            if isinstance(target_datasets, str):
                self._validate_dataset_name_str(target_datasets)
                target_datasets = [target_datasets]
            else:
                assert len(target_datasets) > 0, \
                    "Error: Variable 'target_datasets' cannot be an empty list."
                list(map(lambda dataset_name: self._validate_dataset_name_str(dataset_name), target_datasets))

            self._assert_dataset_available(reference_dataset)
            list(map(lambda dataset_name: self._assert_dataset_available(dataset_name), target_datasets))

            assert reference_dataset not in target_datasets, \
                "Error: Variable 'target_datasets' cannot contain the 'reference_dataset'='{}' " \
                "itself.".format(reference_dataset)

            reference_timestamps = self.get_dataset_timestamps(reference_dataset)
            target_timestamps = list(map(lambda dataset_name: self.get_dataset_timestamps(dataset_name),
                                         target_datasets))

            if len(reference_timestamps) == 0 or any([len(item) == 0 for item in target_timestamps]):
                return False, []

            if modify_hardware_synced_timestamps:
                hardware_synced_datasets = [is_dataset_hardware_synced(self.collection_id, dataset_name)[1] for
                                            dataset_name in [reference_dataset] + target_datasets]
                if sum(hardware_synced_datasets) > 1:
                    current_dataset_names = [reference_dataset] + target_datasets
                    current_dataset_timestamps = [reference_timestamps] + target_timestamps
                    full_dataset_names = self.get_available_dataset_names()
                    hardware_synced_dataset_names = [dataset_name for dataset_name in full_dataset_names if
                                                     is_dataset_hardware_synced(self.collection_id, dataset_name)[1]]
                    full_dataset_timestamps = []
                    for dataset_name in full_dataset_names:
                        if dataset_name in current_dataset_names:
                            full_dataset_timestamps.append(current_dataset_timestamps[
                                                               current_dataset_names.index(dataset_name)])
                        else:
                            full_dataset_timestamps.append(self.get_dataset_timestamps(dataset_name))

                    global_min = np.iinfo(np.integer).max
                    hardware_synced_min = dict(zip(HARDWARE_SYNCED_DATASETS_STRS,
                                                   (np.iinfo(np.integer).max,)*len(HARDWARE_SYNCED_DATASETS_STRS)))

                    for dataset_name, dataset_timestamps in zip(full_dataset_names, full_dataset_timestamps):
                        global_min = min(global_min, min(dataset_timestamps))
                        if dataset_name in hardware_synced_dataset_names:
                            dataset_str = is_dataset_hardware_synced(self.collection_id, dataset_name)[0]
                            hardware_synced_min[dataset_str] = min(hardware_synced_min[dataset_str],
                                                                   min(dataset_timestamps))
                    global_hardware_synced_min = min(hardware_synced_min.values())

                    new_dataset_timestamps = []
                    for dataset_name, dataset_timestamps in zip(full_dataset_names, full_dataset_timestamps):
                        if dataset_name in hardware_synced_dataset_names:
                            dataset_str = is_dataset_hardware_synced(self.collection_id, dataset_name)[0]
                            min_timestamp = hardware_synced_min[dataset_str]
                            new_timestamps = [item - min_timestamp + (global_hardware_synced_min - global_min) for
                                              item in dataset_timestamps]
                            new_dataset_timestamps.append(new_timestamps)
                        else:
                            new_dataset_timestamps.append([item - global_min for item in dataset_timestamps])

                    reference_timestamps = new_dataset_timestamps[full_dataset_names.index(reference_dataset)]
                    target_timestamps = [new_dataset_timestamps[full_dataset_names.index(dataset_name)]
                                         for dataset_name in target_datasets]

            min_reference_timestamp = min(reference_timestamps)
            max_reference_timestamp = max(reference_timestamps)
            min_target_timestamps = list(map(lambda timestamps: min(timestamps), target_timestamps))
            max_target_timestamps = list(map(lambda timestamps: max(timestamps), target_timestamps))
            min_target_timestamp = min(min_target_timestamps)
            max_target_timestamp = max(max_target_timestamps)
            max_min_target_timestamp = max(min_target_timestamps)
            min_max_target_timestamp = min(max_target_timestamps)

            full_overlap_found = False
            if max_reference_timestamp <= min_target_timestamp:
                synced_frames = [[len(reference_timestamps)-1]]
                for _ in range(len(target_datasets)):
                    synced_frames.append([0])
                return full_overlap_found, synced_frames
            elif min_reference_timestamp >= max_target_timestamp:
                synced_frames = [[0]]
                for i in range(len(target_timestamps)):
                    synced_frames.append([len(target_timestamps[i]) - 1])
                return full_overlap_found, synced_frames
            elif max_reference_timestamp <= max_min_target_timestamp:
                frames_idx = [[len(reference_timestamps)-1]]
                for i in range(len(min_target_timestamps)):
                    if max_reference_timestamp <= min_target_timestamps[i]:
                        frames_idx += [[0]]
                    else:
                        frames_idx += [[int(np.argmin(cdist(np.reshape(max_reference_timestamp, (1, 1)),
                                                            np.reshape(target_timestamps[i],
                                                                       (len(target_timestamps[i]), 1))), axis=1))]]
                return full_overlap_found, frames_idx
            elif min_reference_timestamp >= min_max_target_timestamp:
                frames_idx = [[0]]
                for i in range(len(max_target_timestamps)):
                    if min_reference_timestamp >= max_target_timestamps[i]:
                        frames_idx += [[len(target_timestamps[i])-1]]
                    else:
                        frames_idx += [[int(np.argmin(cdist(np.reshape(min_reference_timestamp, (1, 1)),
                                                            np.reshape(target_timestamps[i],
                                                                       (len(target_timestamps[i]), 1))), axis=1))]]
                return full_overlap_found, frames_idx
            else:
                full_overlap_found = True

                reference_idx = [idx for idx, timestamp in enumerate(reference_timestamps)
                                 if max_min_target_timestamp <= timestamp <= min_max_target_timestamp]

                target_idx = []
                for i, target_times in enumerate(target_timestamps):
                    target_idx.append([idx for idx, timestamp in enumerate(target_times)
                                       if max_min_target_timestamp <= timestamp <= min_max_target_timestamp])

                reference_timestamps = reference_timestamps[min(reference_idx):max(reference_idx)+1]
                for i, target_index in enumerate(target_idx):
                    target_timestamps[i] = target_timestamps[i][min(target_index):max(target_index)+1]

                final_target_idx = []
                for i, target_times in enumerate(target_timestamps):

                    min_dist_idx = np.argmin(cdist(np.reshape(reference_timestamps, (len(reference_timestamps), 1)),
                                                   np.reshape(target_times, (len(target_times), 1))), axis=1)

                    final_target_idx.append([target_idx[i][int(ind)] for ind in min_dist_idx])

                return full_overlap_found, [reference_idx] + final_target_idx

        except Exception as e:
            print("{} Exception during extracting synced frames: {}".
                  format(self.__class__.__name__ + ".get_synced_frames()", str(e)))
            return False, []

    def close_reader(self) -> None:
        """
        Closes the reader.

        :return: None
        """
        if self.h5_file is not None:
            if self.h5_file.isopen:
                self.h5_file.close()
                self.h5_file = None

    def get_available_dataset_names(self) -> list:
        """
        Returns all available dataset names within the h5_file.

        :return: List of all available dataset names in the h5_file.
        """
        path = self._get_file_path()
        h5_file = None
        try:
            h5_file = open_h5_file_for_reading(path)
            full_trial_id = self._get_full_trial_id(h5_file, self.trial_id)
            node = h5_file.get_node(where='/', name=full_trial_id)
            dataset_names = list(getattr(node, '_v_links').keys())
        except Exception as e:
            print("{}: Exception during extracting 'dataset_names' from hdf5 file.: \n\t{}".
                  format(self.__class__.__name__ + ".get_available_dataset_names()", str(e)))
            dataset_names = []
        finally:
            close_h5_file(h5_file)

        return dataset_names

    def get_available_dataset_attribute_names(self, dataset_name: str) -> list:
        """
        Returns all available attributes for a specific dataset.

        :param dataset_name: Dataset name of interest (e.g., "BASLER_RGB_BGR")
        :return            : List of all available attribute names for dataset specified by 'dataset_name'.
                             Empty list if no attributes are present. Method raises an Exception if the provided
                             'dataset_name' does not exist.
        """
        self._validate_dataset_name_str(dataset_name)

        dataset_names = self._assert_dataset_available(dataset_name)

        path = self._get_file_path()
        h5_file = None
        try:
            h5_file = open_h5_file_for_reading(path)
            full_trial_id = self._get_full_trial_id(h5_file, self.trial_id)
            node = h5_file.get_node(where='/', name=full_trial_id)
            dataset_path = list(getattr(node, '_v_links').values())[dataset_names.index(dataset_name)].target
            attribute_names = getattr(getattr(h5_file.get_node(where=dataset_path), '_v_attrs'), '_v_attrnamesuser')
        except Exception as e:
            print("{}: Exception during extracting 'attribute_names' from dataset {} of hdf5 file.: \n\t{}".
                  format(self.__class__.__name__ + ".get_available_dataset_attribute_names()", dataset_name, str(e)))
            attribute_names = []
        finally:
            close_h5_file(h5_file)

        return attribute_names

    def get_dataset_attribute(self, dataset_name: str, attribute_name: str):
        """
        Returns an attribute from a dataset.

        :param dataset_name  : Dataset name of interest (e.g., "BASLER_RGB_BGR")
        :param attribute_name: Attribute name of interest (e.g., "timestamps")
        :return              : Value of attribute specified by 'attribute_name' for dataset specified by 'dataset_name'.
                               If the specified dataset or attribute do not exist, the method raises an Exception.
                               The attribute value is returned in its native storage type which is bytes.
        """
        self._validate_dataset_name_str(dataset_name)

        assert isinstance(attribute_name, str), \
            "{} Error: Variable 'attribute_name' must be a " \
            "string.".format(self.__class__.__name__ + ".get_dataset_attribute()")

        dataset_names = self._assert_dataset_available(dataset_name)
        attribute_name = self._assert_attribute_available(dataset_name, attribute_name)

        path = self._get_file_path()
        h5_file = None
        try:
            h5_file = open_h5_file_for_reading(path)
            full_trial_id = self._get_full_trial_id(h5_file, self.trial_id)
            node = h5_file.get_node(where='/', name=full_trial_id)
            dataset_path = list(getattr(node, '_v_links').values())[dataset_names.index(dataset_name)].target
            attribute = h5_file.get_node_attr(where=dataset_path, attrname=attribute_name)
        except Exception as e:
            raise RuntimeError("{} Error: {}".format(self.__class__.__name__ + ".get_dataset_attribute()", str(e)))
        finally:
            close_h5_file(h5_file)

        return attribute

    def get_dataset_wavelength(self, dataset_name: str) -> str:
        """
        Returns the wavelength value for a dataset.

        :param dataset_name: Dataset name of interest (e.g., "BASLER_RGB_BGR")
        :return            : The wavelength attribute of the dataset specified by 'dataset_name' as a string.
                             If no wavelength is associated with the dataset, it returns an empty string.
                             Raises an Exception if the dataset does not exist.
        """
        try:
            wavelength = self.get_dataset_attribute(dataset_name=dataset_name, attribute_name="wavelength")
            wavelength = wavelength.decode('utf-8')
            wavelength = ''.join([item for item in wavelength if (item.isalnum() or item == "_")])
        except AttributeError:
            return ''
        except Exception as e:
            raise RuntimeError("{} Error: {}".format(self.__class__.__name__ + ".get_dataset_wavelength()", str(e)))

        return wavelength

    def get_dataset_timestamps(self, dataset_name: str) -> list:
        """
        Returns the timestamps for a dataset.

        :param dataset_name: dataset name of interest (e.g., "BASLER_RGB_BGR")
        :return            : the timestamps attribute of the dataset specified by 'dataset_name' as a list of integers.
                             If no timestamps are available, it returns an empty list. Raises an exception if the
                             dataset does not exist.
        """
        try:
            timestamps = self.get_dataset_attribute(dataset_name=dataset_name, attribute_name='timestamps')
            timestamps = timestamps.decode('utf-8').split(',')
            timestamps = list(map(lambda entry: int(''.join([item for item in entry if item.isdigit()])), timestamps))
        except AttributeError:
            return []
        except Exception as e:
            raise RuntimeError("{} Error: {}".format(self.__class__.__name__ + ".get_dataset_timestamps()", str(e)))

        return timestamps

    def _get_modality(self) -> (None, str):

        modality = None
        path = self._get_file_path()
        h5_file = None
        try:
            h5_file = open_h5_file_for_reading(path)
            attr_names = getattr(getattr(h5_file.root, '_v_attrs'), '_v_attrnamesuser')
            modality_attr_name = [name for name in attr_names if name.upper() == 'MODALITY']

            if len(modality_attr_name) == 1:
                modality = h5_file.get_node_attr(where='/', attrname=modality_attr_name[0]).upper()
            else:
                modality = None
        except (AttributeError, KeyError):
            pass
        except Exception as e:
            print("{}: Exception during extracting 'modality' from hdf5 file.: \n\t{}".
                  format(self.__class__.__name__, str(e)))
        finally:
            close_h5_file(h5_file)

        return modality

    def get_collection_id(self) -> (None, str):
        """
        Returns the collection_id for the underlying h5_file.

        :return: A string uniquely identifying the collection id of the reader (one of 'ST1', 'ST2', 'GCT1', 'ST3',
                 'GCT2'). Can be used for processing data from different collections in a custom way.
        """

        collection_id_mapping = dict()

        path = self._get_file_path()
        date_key = None
        try:
            with tables.open_file(path, 'r') as h5_file:
                attr_names = getattr(getattr(h5_file.root, '_v_attrs'), '_v_attrnamesuser')
                date_attr_name = [name for name in PadisiH5Reader.DATE_ATTR_STRS if name in attr_names]

                if len(date_attr_name) == 1:
                    collection_id_mapping = PadisiH5Reader.DATE_COLLECTION_ID_MAPPING
                    date_key = str(h5_file.get_node_attr(where='/', attrname=date_attr_name[0]))
                    date_key = date_key[:10]
        except (AttributeError, KeyError):
            pass
        except Exception as e:
            print("{}: Exception during extracting 'collection_id' from hdf5 file.: \n\t{}".
                  format(self.__class__.__name__, str(e)))

        collection_id = None
        if date_key is not None:
            try:
                date_collection_keys = list(collection_id_mapping.keys())
                date_collection_keys.sort()

                if date_key >= date_collection_keys[-1]:
                    collection_id = collection_id_mapping[date_collection_keys[-1]]
                elif date_key < date_collection_keys[0]:
                    collection_id = None
                else:
                    if date_key in date_collection_keys:
                        collection_id = collection_id_mapping[date_key]
                    else:
                        index = 0
                        while date_key > date_collection_keys[index]:
                            index += 1
                        collection_id = collection_id_mapping[date_collection_keys[index-1]]
            except Exception as e:
                print("{}: Exception during extracting 'collection_id' from hdf5 file.: \n\t{}".
                      format(self.__class__.__name__, str(e)))
        else:
            collection_id = None

        return collection_id

    @staticmethod
    def get_collection_id_from_file(file_path: str) -> (None, str):
        """
        :return: a string uniquely identifying the collection id of the reader (one of 'ST1', 'ST2', 'GCT1', 'ST3',
                 'GCT2'). Can be used for processing data from different collections in a custom way.
        """

        assert isinstance(file_path, str), \
            "Exception during extracting 'collection_id' from file. The provided 'file_path' must be a string."

        assert os.path.exists(file_path) and os.path.isfile(file_path) and os.path.splitext(file_path)[1] == '.h5', \
            "Exception during extracting 'collection_id' from file. The provided file '{}' must be an" \
            "existing '.h5' file.".format(file_path)

        collection_id_mapping = dict()
        date_key = None
        try:
            with tables.open_file(file_path, 'r') as h5_file:
                attr_names = getattr(getattr(h5_file.root, '_v_attrs'), '_v_attrnamesuser')
                date_attr_name = [name for name in PadisiH5Reader.DATE_ATTR_STRS if name in attr_names]

                if len(date_attr_name) == 1:
                    collection_id_mapping = PadisiH5Reader.DATE_COLLECTION_ID_MAPPING
                    date_key = str(h5_file.get_node_attr(where='/', attrname=date_attr_name[0]))
                    date_key = date_key[:10]
        except (AttributeError, KeyError):
            pass
        except Exception as e:
            print("Exception during extracting 'collection_id' from hdf5 file '{}'.: \n\t{}".
                  format(file_path, str(e)))

        collection_id = None
        if date_key is not None:
            try:
                date_collection_keys = list(collection_id_mapping.keys())
                date_collection_keys.sort()

                if date_key >= date_collection_keys[-1]:
                    collection_id = collection_id_mapping[date_collection_keys[-1]]
                elif date_key < date_collection_keys[0]:
                    collection_id = None
                else:
                    if date_key in date_collection_keys:
                        collection_id = collection_id_mapping[date_key]
                    else:
                        index = 0
                        while date_key > date_collection_keys[index]:
                            index += 1
                        collection_id = collection_id_mapping[date_collection_keys[index-1]]
            except Exception as e:
                print("Exception during extracting 'collection_id' from hdf5 file '{}'.: \n\t{}".
                      format(file_path, str(e)))
        else:
            collection_id = None

        return collection_id

    def _get_file_path(self) -> (None, str):
        if os.path.isdir(self.db_path):
            return os.path.join(self.db_path, self.transaction_id + ".h5")
        else:
            return self.db_path

    def _get_full_trial_id(self, h5_file: tables.file.File, trial_id: str) -> str:
        try:
            group_list = list(map(lambda item: getattr(item, '_v_name'),
                                  h5_file.list_nodes(where='/', classname='Group')))
            full_trial_id = group_list[[trial_id[:-1] in item for item in group_list].index(True)]
        except Exception as e:
            raise RuntimeError("{} Error: {}".format(self.__class__.__name__, str(e)))

        return full_trial_id

    def _assert_dataset_available(self, dataset_name: str) -> list:

        dataset_names = self.get_available_dataset_names()
        assert (dataset_name in dataset_names), \
            "{} Error: Dataset '{}' does not exist in hdf5 file. Available dataset names are: " \
            "{}.".format(self.__class__.__name__ + "._assert_dataset_available()", dataset_name,
                         ', '.join(["'" + item + "'" for item in dataset_names]))

        return dataset_names

    def _assert_attribute_available(self, dataset_name: str, attribute_name: str) -> str:

        attribute_names = self.get_available_dataset_attribute_names(dataset_name)
        attribute_names_lower = list(map(lambda item: item.lower(), attribute_names))
        try:
            assert attribute_name.lower() in attribute_names_lower, \
                "{} Error: Attribute '{}' does not exist in dataset {} of hdf5 file. " \
                "Available attribute names are: " \
                "{}.".format(self.__class__.__name__ + "._assert_attribute_available()", attribute_name, dataset_name,
                             ', '.join(["'" + item + "'" for item in attribute_names]))
        except AssertionError as e:
            raise AttributeError(str(e))

        return attribute_names[attribute_names_lower.index(attribute_name.lower())]

    def _validate_string_variable(self, var_val: str, var_name: str) -> None:
        assert isinstance(var_name, str), "Variable 'var_name' must be a string."
        assert isinstance(var_val, str), "{} Error: Property {} must be a string.". \
            format(self.__class__.__name__, var_name)

    def _validate_dataset_name_str(self, dataset_name: str) -> None:
        assert isinstance(dataset_name, str), \
            "{} Error: Variable 'dataset_name' must be a " \
            "string.".format(self.__class__.__name__ + "._validate_dataset_name_str()")
        assert len(dataset_name) > 0, \
            "{} Error: Variable 'dataset_name'='{}' cannot be an empty " \
            "string.".format(self.__class__.__name__ + "._validate_dataset_name_str()", dataset_name)
        assert not dataset_name.startswith('/'), \
            "{} Error: Variable 'dataset_name'='{}' cannot start with " \
            "'/'.".format(self.__class__.__name__ + "._validate_dataset_name_str()", dataset_name)

    def _verify_existence(self) -> None:
        assert self.db_path is not None, \
            "{} Error: Property 'db_path' cannot be None if property 'check_existence' is " \
            "True.".format(self.__class__.__name__ + "._verify_existence()")
        if os.path.isdir(self.db_path):
            path_transaction_level = os.path.join(self.db_path, self.transaction_id + ".h5")

            if not os.path.isfile(path_transaction_level):
                raise RuntimeError("{}._verify_existence() Error: No transaction files "
                                   "found.".format(self.__class__.__name__))

            try:
                with tables.open_file(path_transaction_level) as h5_file:
                    try:
                        successful_trials = h5_file.get_node_attr(where='/', attrname='successful_trials')
                        if self.trial_id[:-1] in successful_trials:
                            return
                    except AttributeError as e:
                        print("{} Error: {}. Attribute 'successful_trials' not found. Attempt to verify "
                              "'trial_id'".format(self.__class__.__name__, str(e)))

                    full_trial_id = self._get_full_trial_id(h5_file, self.trial_id)
                    h5_file.get_node(where='/', name=full_trial_id)
            except Exception as e:
                raise Exception("{}._verify_existence Error: {}".format(self.__class__.__name__, str(e)))
        else:
            try:
                h5_file = tables.open_file(self.db_path)
                h5_file.close()
            except Exception as e:
                raise Exception("{}._verify_existence Error: {}".format(self.__class__.__name__, str(e)))

    @property
    def db_path(self) -> (None, str):
        return self._db_path

    @db_path.setter
    def db_path(self, db_path: (None, str)) -> None:
        if db_path is not None:
            assert isinstance(db_path, str) and len(db_path) > 0, \
                "{} Error: Property 'db_path' must be a non-empty string.".format(self.__class__.__name__)
        self._db_path = db_path

    @property
    def transaction_id(self) -> str:
        return self._transaction_id

    @transaction_id.setter
    def transaction_id(self, transaction_id: str) -> None:
        self._validate_string_variable(transaction_id, 'transaction_id')
        self._transaction_id = transaction_id

    @property
    def trial_id(self) -> str:
        return self._trial_id

    @trial_id.setter
    def trial_id(self, trial_id: str) -> None:
        self._validate_string_variable(trial_id, 'trial_id')
        assert len(trial_id) > 1, \
            "{} Error: Property 'trial_id' must be a string with at least " \
            "two characters.".format(self.__class__.__name__)
        self._trial_id = trial_id

    @property
    def pai(self) -> str:
        return self._pai

    @pai.setter
    def pai(self, pai: str) -> None:
        self._validate_string_variable(pai, 'pai')
        assert any(list(map(lambda str_begin: pai.startswith(str_begin), PadisiH5Reader.VALID_PAI_START_CHARS))), \
            "{} Error: Property 'pai'={} is invalid. It must start with any of {}.".format(
                self.__class__.__name__, pai,
                ', '.join(["'" + item + "'" for item in PadisiH5Reader.VALID_PAI_START_CHARS]))
        self._pai = pai

    @property
    def trial_name(self) -> (None, str):
        return self._trial_name

    @trial_name.setter
    def trial_name(self, trial_name: (None, str)) -> None:
        if trial_name is not None:
            self._validate_string_variable(trial_name, 'trial_name')
        self._trial_name = trial_name

    @property
    def data_extractor(self) -> (None, ReaderDataExtractor, tuple, list):
        return self._data_extractor

    @data_extractor.setter
    def data_extractor(self, data_extractor: (None, ReaderDataExtractor, tuple, list)) -> None:
        if data_extractor is not None:
            assert isinstance(data_extractor, (ReaderDataExtractor, tuple, list)), \
                "{} Error: Property 'data_extractor' must be an 'ReaderDataExtractor' object, a tuple, or a " \
                "list.".format(self.__class__.__name__)
            if isinstance(data_extractor, (tuple, list)):
                assert len(data_extractor) > 0 and all(
                    list(map(lambda item: isinstance(item, ReaderDataExtractor), data_extractor))), \
                    "{} Error: If property 'data_extractor' is a tuple or list, it must contain " \
                    "'ReaderDataExtractor' objects.".format(self.__class__.__name__)
                if len(data_extractor) == 1:
                    data_extractor = data_extractor[0]
        self._data_extractor = data_extractor

    @property
    def h5_file(self) -> (None, tables.file.File):
        return self._h5_file

    @h5_file.setter
    def h5_file(self, h5_file: (None, tables.file.File)):
        if h5_file is not None:
            assert isinstance(h5_file, tables.file.File), \
                "{} Error: Property 'h5_file' must be a 'tables.file.File' " \
                "object.".format(self.__class__.__name__)
        self._h5_file = h5_file


# Base class extending the base class ReaderPropertyExtractor for class PadisiH5Reader

# Available methods:
#   .extract_property()
class PadisiH5ReaderPropertyExtractor(ReaderPropertyExtractor):

    @abstractmethod
    def _validate_property_value(self, property_value) -> None:
        """
        Used to validate a property value returned by the property extractor.

        :param property_value: Property value returned by the property extractor.
        :return              : None
        """
        pass

    @abstractmethod
    def _extract_property_value(self, padisi_h5_reader: PadisiH5Reader):
        """
        Used to extract a property value from an PadisiH5Reader.

        :param padisi_h5_reader: PadisiH5Reader to extract a property from.
        :return                : Property value.
        """
        pass

    def extract_property(self, data_reader: DataReader):
        """
        Extracts a property value from a PadisiH5Reader and validates it

        :param data_reader: PadisiH5Reader to extract a property from.
        :return           : Property value.
        """
        assert isinstance(data_reader, PadisiH5Reader), \
            "{} Error: Variable 'data_reader' must be an 'PadisiH5Reader' " \
            "object.".format(self.__class__.__name__ + ".extract_property()")

        property_value = self._extract_property_value(data_reader)

        self._validate_property_value(property_value)

        return property_value


# Base class extending the PadisiH5ReaderPropertyExtractor for properties of type list.
class PadisiH5ReaderListPropertyExtractor(PadisiH5ReaderPropertyExtractor):

    @abstractmethod
    def _get_list_values_validation_funcs(self) -> (None, Callable, list):
        """
        Used to validate the entries of the returned list property.

        :return: None     -> No validation is performed.
                 Callable -> Each entry of the list property is validated using the Callable.
                 list     -> Each entry of the list property is validated using a different validation method
                             in the list.
        """
        pass

    @abstractmethod
    def _extract_property_list_value(self, padisi_h5_reader: PadisiH5Reader) -> list:
        """
        Used to extract a list property value from an PadisiH5Reader.

        :param padisi_h5_reader: PadisiH5Reader to extract a list property from.
        :return                : List property value.
        """
        pass

    def _extract_property_value(self, padisi_h5_reader: PadisiH5Reader) -> list:
        return self._extract_property_list_value(padisi_h5_reader)

    def _validate_property_value(self, property_value: list) -> None:
        assert isinstance(property_value, list), \
            "{} Error: Property '{}' returned by method '_extract_property_list_value()' must return a " \
            "list.".format(self.__class__.__name__, self.property_name)

        validation_funcs = self._get_list_values_validation_funcs()

        if validation_funcs is None:
            return

        if validation_funcs is None or callable(validation_funcs):
            validation_funcs = [validation_funcs]
        assert all(list(map(lambda item: item is None or callable(item), validation_funcs))), \
            "{} Error: Method '_get_list_values_validation_funcs()' must return None or a callable object or a " \
            "list containing either entries which are None or callable " \
            "objects.".format(self.__class__.__name__ + "._validate_property_value()")

        if len(validation_funcs) == 1:
            validation_funcs = validation_funcs*len(property_value)
        else:
            assert len(validation_funcs) == len(property_value), \
                "{} Error: Property '{}' has incorrect number of entries: {} but expected " \
                "{}.".format(self.__class__.__name__ + "._validate_property_value()", self.property_name,
                             len(property_value), len(validation_funcs))

        try:
            for validation_func, value in zip(validation_funcs, property_value):
                if validation_func is not None:
                    validation_func(value)
        except Exception:
            raise RuntimeError("{} Error: Exception while validating entries of property '{}'".
                               format(self.__class__.__name__ + "._validate_property_value()", self.property_name))


# Base class extending the PadisiH5ReaderPropertyExtractor for properties of type dict.
class PadisiH5ReaderDictPropertyExtractor(PadisiH5ReaderPropertyExtractor):

    @abstractmethod
    def _get_dict_entries_validation_dictionary(self) -> (None, Callable, dict):
        """
        Used to validate the entries of the returned dictionary property.

        :return: None     -> No validation is performed.
                 Callable -> Each value of the dictionary property is validated using the Callable.
                 dict     -> Each entry of the dictionary property is validated using a different validation method
                             in the dictionary.
        """
        pass

    @abstractmethod
    def _extract_dictionary_value(self, padisi_h5_reader: PadisiH5Reader) -> dict:
        """
        Used to extract a dictionary property value from an PadisiH5Reader.

        :param padisi_h5_reader: PadisiH5Reader to extract a list property from.
        :return                : Dictionary property value.
        """
        pass

    def _extract_property_value(self, padisi_h5_reader: PadisiH5Reader) -> dict:
        return self._extract_dictionary_value(padisi_h5_reader)

    def _validate_property_value(self, property_value) -> None:
        assert isinstance(property_value, dict), \
            "{} Error: Method '_extract_dictionary_value()' must return a " \
            "dictionary.".format(self.__class__.__name__ + "._validate_property_value()")

        key_value_validation_dictionary = self._get_dict_entries_validation_dictionary()

        if key_value_validation_dictionary is None:
            return

        assert isinstance(key_value_validation_dictionary, dict) or callable(key_value_validation_dictionary), \
            "{} Error: Method '_get_key_value_validation_dictionary()' must return a dictionary or a callable " \
            "object.".format(self.__class__.__name__ + "._validate_property_value()")

        if isinstance(key_value_validation_dictionary, dict):
            if len(key_value_validation_dictionary) == 0:
                return
            assert all(list(map(lambda item: item is None or callable(item),
                                list(key_value_validation_dictionary.values())))), \
                "{} Error: Dictionary returned by method '_get_key_value_validation_dictionary()' " \
                "contains invalid values. All values must be either None or a callable " \
                "object.".format(self.__class__.__name__ + "._validate_property_value()")

            missing_keys = [item for item in key_value_validation_dictionary.keys()
                            if item not in property_value.keys()]
            assert len(missing_keys) == 0, \
                "{} Error: Keys '{}' are missing from variable 'property_" \
                "value'.".format(self.__class__.__name__ + "._validate_property_value()",
                                 ', '.join(["'" + item + "'" for item in missing_keys]))

            invalid_keys = [item for item in property_value.keys() if
                            item not in key_value_validation_dictionary.keys()]
            assert len(invalid_keys) == 0, \
                "{} Error: Property '{}' contains invalid keys: [{}]. Valid keys are: " \
                "[{}]".format(self.__class__.__name__ + "._validate_property_value()", self.property_name,
                              ', '.join(["'" + item + "'" for item in invalid_keys]),
                              ', '.join(["'" + item + "'" for item in key_value_validation_dictionary.keys()]))
        else:
            validation_funcs = [key_value_validation_dictionary]*len(property_value)
            key_value_validation_dictionary = dict(zip(property_value.keys(), validation_funcs))

        try:
            for key, validation_func in key_value_validation_dictionary.items():
                if validation_func is not None:
                    validation_func(property_value[key])
        except Exception:
            raise RuntimeError("{} Error: Exception while validating entries of property '{}'.".
                               format(self.__class__.__name__ + "._validate_property_value()", self.property_name))


# Base class extending the base class ReaderDataExtractor for PadisiH5Reader.

# Available methods:
#    .extract_data()
class PadisiH5ReaderDataExtractor(ReaderDataExtractor):

    def __init__(self, property_extractors: (None, list, PadisiH5ReaderPropertyExtractor,
                                             ReaderMultiPropertyExtractor) = None):
        super(PadisiH5ReaderDataExtractor, self).__init__(property_extractors)
        if self.property_extractors is not None:
            assert all(list(map(lambda item: isinstance(item, PadisiH5ReaderPropertyExtractor),
                                self.property_extractors.property_extractors.values()))), \
                "{} Error: Property 'property_extractors' must be a 'ReaderMultiPropertyExtractor' containing" \
                "'PadisiH5ReaderPropertyExtractor' objects.".format(self.__class__.__name__)

    @abstractmethod
    def get_property_extractors_validation_dictionary(self) -> (None, Callable, dict):
        """
        Used to validate the property extractors provided as input.

        :return: None     -> No validation is performed.
                 Callable -> Each property extractor is validated using the Callable.
                 dict     -> Each property extractor is validated using the entries of the dictionary.
        """
        pass

    @abstractmethod
    def _extract_padisi_h5_data(self, padisi_h5_reader: PadisiH5Reader):
        """
        Method for extracting data from an PadisiH5Reader.

        :param padisi_h5_reader: PadisiH5Reader to extract data from.
        :return                : Extracted data
        """
        pass

    def extract_data(self, data_reader: PadisiH5Reader) -> np.ndarray:
        """
        Extracts data from an PadisiH5Reader after verifying the the provided object is indeed an PadisiH5Reader.

        :param data_reader: PadisiH5Reader to extract data from.
        :return           : Extracted data.
        """
        assert isinstance(data_reader, PadisiH5Reader), \
            "{} Error: Variable 'data_reader' must be an 'PadisiH5Reader' object.".format(self.__class__.__name__)

        return self._extract_padisi_h5_data(data_reader)


# Class extending class PadisiH5Reader to multi-label data. Labels must be integers greater or equal to 0.
class PadisiH5MultiLabelReader(PadisiH5Reader):

    def __init__(self, db_path: (None, str), transaction_id: str, trial_id: str, pai: str,
                 ground_truth, modality: str, trial_name: (None, str),
                 data_extractor: (None, ReaderDataExtractor, tuple, list), check_existence: bool = True):
        super(PadisiH5MultiLabelReader, self).__init__(db_path=db_path, transaction_id=transaction_id,
                                                       trial_id=trial_id,
                                                       pai=pai, ground_truth=ground_truth, modality=modality,
                                                       trial_name=trial_name, data_extractor=data_extractor,
                                                       check_existence=check_existence)

        # Make sure the name is PadisiH5Reader to not affect .h5 file id when only the labels change.
        self.__class__.__name__ = self.__class__.__name__.replace('MultiLabel', '')

    @property
    def ground_truth(self) -> (None, int):
        return self._ground_truth

    @ground_truth.setter
    def ground_truth(self, ground_truth: int) -> None:
        assert isinstance(ground_truth, int) and ground_truth >= 0, \
            "{} Error: Property 'ground_truth' must be a non-negative " \
            "integer.".format(self.__class__.__name__)
        self._ground_truth = ground_truth


# Class extending the base class BiometricMultiLabelData to PADISI data.
class PadisiH5MultiLabelData(BiometricMultiLabelData):

    DEFAULT_PAI_CODE_LABEL_MAPPINGS = \
        {'FINGER': {'m000': 0,
                    'm401': 1,
                    'm402': 2,
                    'm403': 3,
                    'm404': 4,
                    'm405': 5,
                    'm406': 6,
                    'm407': 7,
                    'm408': 8,
                    'm409': 9,
                    'm410': 10,
                    'm411': 11,
                    'm412': 12,
                    'm413': 13,
                    'm414': 14,
                    'm415': 15,
                    'm416': 16,
                    'm417': 17,
                    'm418': 18,
                    'm419': 19,
                    'm420': 20,
                    'm421': 21,
                    'm422': 22,
                    'm424': 23,
                    'm425': 24,
                    'm426': 25,
                    'm428': 26,
                    'm429': 27},
         'FACE': {'m000': 0,
                  'm501': 1,
                  'm502': 2,
                  'm503': 3,
                  'm504': 4,
                  'm505': 5,
                  'm506': 6,
                  'm507': 7,
                  'm508': 8,
                  'm509': 9}
         }

    def __init__(self, modality: str, pai_code_label_mapping: (None, dict) = None):
        self.modality = modality
        super(PadisiH5MultiLabelData, self).__init__(pai_code_label_mapping)

    def get_default_pai_code_label_mapping(self) -> dict:
        assert self.modality.upper() in PadisiH5MultiLabelData.DEFAULT_PAI_CODE_LABEL_MAPPINGS, \
            "{} Error: 'DEFAULT_PAI_CODE_LABEL_MAPPINGS' does not contain key " \
            "'{}'.".format(self.__class__.__name__, self.modality.upper())
        return PadisiH5MultiLabelData.DEFAULT_PAI_CODE_LABEL_MAPPINGS[self.modality.upper()]


# Abstract class PadisiH5Dataset extending the base class CsvBiometricData to PADISI data.
#
# Abstract methods:
#    .get_data_extractor()
#
# Available methods:
#    .get_partition()
#    .create_online_reader()
#    .create_reader()
#    .is_pa()
class PadisiH5Dataset(CsvBiometricDataset):
    def __init__(self, db_path: (None, str), ground_truth_path: str,
                 dataset_partition_path: str, modality: str,
                 data_extractor: (None, PadisiH5ReaderDataExtractor, tuple, list) = None,
                 needed_ground_truth_entries: (None, tuple) = ('ground_truth', 'modality'),
                 optional_ground_truth_entries: (None, tuple) = ('trial_name', 'relative_path'),
                 multi_label: bool = False, pai_code_label_mapping: (None, dict) = None,
                 check_file_existence: bool = False):
        super(PadisiH5Dataset, self).__init__(db_path, ground_truth_path, dataset_partition_path, modality,
                                              needed_ground_truth_entries,
                                              optional_ground_truth_entries)
        self.multi_label = multi_label
        if self.multi_label:
            self.multi_label_data = PadisiH5MultiLabelData(pai_code_label_mapping, self.modality)
        else:
            self.multi_label_data = None
        self._reader_inputs = {'data_extractor': data_extractor, 'check_existence': check_file_existence}

    def create_reader(self, reader_entries: dict) -> (PadisiH5Reader, PadisiH5MultiLabelReader):
        """
        Creates an PadisiH5Reader based on the provided entries in the input .csv files through the input
        dictionary reader_entries.

        :param reader_entries: Dictionary with key/values extracted from the input .csv files.
        :return              : A PadisiH5Reader or PadisiH5MultiLabelReader object.
        """

        missing_entries = [item for item in PadisiH5Reader.REQUIRED_CSV_READER_ENTRIES if
                           item not in reader_entries.keys()]
        assert len(missing_entries) == 0, "{} Error: Variable 'reader_entries' must contain entries: {}".format(
            self.__class__.__name__, ', '.join(["'" + item + "'" for item in missing_entries]))

        reader_db_path = self.db_path
        if reader_entries['relative_path'] is not None:
            assert isinstance(reader_entries['relative_path'], str), \
                "{} Error: Entry 'reader_entries['relative_path']' must be a " \
                "string.".format(self.__class__.__name__ + ".create_reader()")
            if len(reader_entries['relative_path']) > 0:
                if reader_db_path is not None:
                    reader_db_path = os.path.join(reader_db_path, reader_entries['relative_path'])
                else:
                    reader_db_path = reader_entries['relative_path']

        if self.multi_label_data is not None:
            return PadisiH5MultiLabelReader(reader_db_path, reader_entries['transaction_id'],
                                            reader_entries['trial_id'],
                                            reader_entries['ground_truth'],
                                            self.multi_label_data.get_pai_label(reader_entries['ground_truth']),
                                            reader_entries['modality'],
                                            reader_entries['trial_name'],
                                            **self._reader_inputs)
        else:
            return PadisiH5Reader(reader_db_path, reader_entries['transaction_id'],
                                  reader_entries['trial_id'],
                                  reader_entries['ground_truth'],
                                  self.is_pa(reader_entries['ground_truth'], reader_entries['modality']),
                                  reader_entries['modality'],
                                  reader_entries['trial_name'],
                                  **self._reader_inputs)

    def is_pa(self, pai: str, modality: (None, str) = None) -> bool:
        """
        :param pai     : Ground truth code of a sample.
        :param modality: Biometric modality. One of ['FACE', 'FINGER', 'IRIS'] or None.
             (Optional)  Default is None. If None, the modality provided as an input is used. If no modality
                         is provided as input, it raises an Exception.
        :return        : True if ground truth code corresponds to a PAI, else False.
        """
        assert isinstance(pai, str), "Variable 'pai' must be a string."
        if modality is None:
            modality = self.modality
            assert modality is not None, \
                "{} Error: 'is_pa' method cannot be used if 'modality' is 'None'.".format(self.__class__.__name__)
        else:
            assert isinstance(modality, str), "Variable 'modality' must be a string."

        return is_padisi_gt_pa(pai, modality)

    @property
    def multi_label(self) -> bool:
        return self._multi_label

    @multi_label.setter
    def multi_label(self, multi_label: bool) -> None:
        assert isinstance(multi_label, bool), "{} Error: Property 'multi_label' must be a bool.". \
            format(self.__class__.__name__)
        self._multi_label = multi_label

    @property
    def multi_label_data(self) -> BiometricMultiLabelData:
        return self._multi_label_data

    @multi_label_data.setter
    def multi_label_data(self, multi_label_data: (None, BiometricMultiLabelData)) -> None:
        if multi_label_data is not None:
            assert isinstance(multi_label_data, BiometricMultiLabelData), \
                "{} Error: Property 'multi_label_data' must be an instance of " \
                "'BiometricMultiLabelData'.".format(self.__class__.__name__)
        self._multi_label_data = multi_label_data


# Class PadisiH5MultiDataset extending the base class MultiDataset for objects of class PadisiH5Dataset.
class PadisiH5MultiDataset(MultiDataset):

    @property
    def datasets(self) -> list:
        return self._datasets

    @datasets.setter
    def datasets(self, padisi_datasets: (tuple, list)) -> None:
        assert isinstance(padisi_datasets, (tuple, list)) and len(padisi_datasets) > 0 and all(
            list(map(lambda item: isinstance(item, PadisiH5Dataset), padisi_datasets))), \
            "{} Error: Input 'padisi_datasets' must be a tuple or list of 'PadisiH5Dataset' " \
            "objects.".format(self.__class__.__name__)
        self._datasets = padisi_datasets
