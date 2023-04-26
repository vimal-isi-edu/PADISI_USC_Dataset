import inspect
import os

import joblib
import numpy as np

try:
    import torch
    from torch.utils.data.dataloader import DataLoader
    from torch.utils.data.dataset import TensorDataset, Dataset, T_co
except ModuleNotFoundError:
    torch = None
    DataLoader = None
    TensorDataset = None


class ReaderDataset(Dataset):

    def __init__(self, readers: list) -> None:
        self.readers = readers

    def __getitem__(self, index) -> T_co:
        return np.squeeze(self.readers[index].get_data()), int(self.readers[index].get_ground_truth())

    def __len__(self):
        return len(self.readers)


def load_preprocessed_data(file_path: str) -> dict:
    """
    :param file_path: Path to the file name or folder of stored pre-processed data.
    :return         : Dictionary of loaded data or mapping between id and data files.
    """

    error_str = inspect.currentframe().f_code.co_name + '()'
    assert isinstance(file_path, str) and len(file_path) > 0, \
        "{} Error: Variable 'file_path' must be a non-empty string.".format(error_str)

    assert os.path.isdir(file_path) or os.path.isfile(file_path) and os.path.splitext(file_path)[1] == '.bz1', \
        "{} Error: Variable 'file_path' must be an existing directory or a path to an existing '.bz2' " \
        "file".format(error_str)

    if os.path.isdir(file_path):
        files = [os.path.join(file_path, file) for file in os.listdir(file_path) if os.path.splitext(file)[1] == '.npz']
        data_dict = dict(zip([os.path.splitext(os.path.split(file)[1])[0] for file in files], files))
    else:
        print('Loading pre-processed data... This might take a while...')
        data_dict = joblib.load(file_path)
        print('Data loading successful.')
    return data_dict


def convert_partition_to_loader(partition: list, partition_name: str,
                                batch_size: int) -> (None, DataLoader):
    """
    :param partition     : List of data readers allowing reading data for the current partition.
    :param partition_name: String of the partition name. One of ('train', 'valid', 'test')
    :param batch_size    : Batch size for the DataLoader.
    :return              : None, if partition is an empty list or DataLoader object using the provided batch_size.
                           Note that for 'train' and 'valid' partitions shuffling is set to True while for the 'test'
                           partition to False.
    """

    assert torch is not None, \
        "{} Error: PyTorch is required. Install it from https://pytorch.org/.".format('convert_partition_to_loader()')

    if len(partition) == 0:
        return None

    multi_spectral_data = getattr(partition[0].data_extractor, '_multi_spectral_data')

    if multi_spectral_data:
        # Create dataset
        dataset = ReaderDataset(partition)
    else:
        # Create tensors
        data = torch.as_tensor(np.concatenate([reader.get_data() for reader in partition])).float()
        labels = torch.as_tensor([reader.get_ground_truth() for reader in partition]).long()

        # Create dataset
        dataset = TensorDataset(data, labels)
    # Create loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False if partition_name == 'test' else True)

    return loader
