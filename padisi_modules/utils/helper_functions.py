import os

import joblib
import numpy as np
try:
    import torch
    from torch.utils.data.dataloader import DataLoader
    from torch.utils.data.dataset import TensorDataset
except ModuleNotFoundError:
    torch = None
    DataLoader = None
    TensorDataset = None


def load_preprocessed_data(file_name: str) -> dict:
    """
    :param file_name: Path to the file name of stored pre-processed data.
    :return         : Dictionary of loaded data.
    """

    assert isinstance(file_name, str) and len(file_name) > 0 and os.path.isfile(file_name) and \
        os.path.splitext(file_name)[1] == ".bz2", \
        "{} Error: Variable 'file_name' must be a path to an existing '.bz2' file.".format('load_preprocessed_data()')

    print('Loading pre-processed data... This might take a while...')
    data_dict = joblib.load(file_name)
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

    # Create tensors
    data = torch.as_tensor(np.concatenate([reader.get_data() for reader in partition])).float()
    labels = torch.as_tensor([reader.get_ground_truth() for reader in partition]).long()

    # Create loader
    tensor_dataset = TensorDataset(data, labels)
    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False if partition_name == 'test' else True)

    return loader
