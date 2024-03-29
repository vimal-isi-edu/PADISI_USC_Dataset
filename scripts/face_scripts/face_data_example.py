import argparse
import os

from padisi_modules.dataset_utils.face_datasets.padisi_face_datasets_helpers import create_padisi_face_dataset
from padisi_modules.utils.helper_functions import load_preprocessed_data, convert_partition_to_loader

try:
    import torch
except ModuleNotFoundError:
    torch = None


# Example use when using pre-processed COLOR data:
#   The code assumes that:
#   -- The pre-processed data have been downloaded and are under data/face_data/preprocessed/
#      padisi_USC_FACE_preprocessed.bz2 - Provide a different path otherwise through the (-data_path) flag.
#   -- We select the 3fold_part0 partition (see all partitions under data/face_partitions)
#   -- We select COLOR data - This is the only option for these data.
#
#   conda activate padisi
#   python face_data_example.py
#          -dbp ../../data/face_partitions/padisi_USC_FACE_dataset_partition_3folds_part0.csv
#          -extractor_id COLOR

# Example use when using pre-processed multi-channel data:
#   The code assumes that:
#   -- The pre-processed multi-channel data have been downloaded, unzipped and are under the
#      data/face_data/preprocessed/multi_channel folder
#      - Provide a different path otherwise through the (-data_path) flag.
#   -- We select the 3fold_part0 partition (see all partitions under data/face_partitions)
#   -- We select COLOR data - The user can select any combination of COLOR, NIR, DEPTH, THERMAL, NIRL, NIRR, SWIR,
#      separated by underscores. Regardless of the order provided, the data will always be returned in the
#      aforementioned order.  For example -extractor_id COLOR_SWIR and -extractor_id SWIR_COLOR will always
#      return channels 0-2, 16-21 in sorted order:
#
#   conda activate padisi
#   python face_data_example.py
#          -data_path ../../data/face_data/preprocessed/multi_channel
#          -dbp ../../data/face_partitions/padisi_USC_FACE_dataset_partition_3folds_part0.csv
#          -extractor_id COLOR

def main():
    parser = argparse.ArgumentParser(description='Arguments for testing the creation of a PyTorch loader.')
    parser.add_argument('-data_path', dest='data_path', type=str,
                        default=os.path.join('..', '..', 'data', 'face_data', 'preprocessed',
                                             'padisi_USC_FACE_preprocessed.bz2'),
                        help='Path to the .bz2 file containing the pre-processed data.')
    parser.add_argument('-gt', dest='ground_truth_file', type=str,
                        default=os.path.join('..', '..', 'data', 'face_partitions',
                                             'padisi_USC_FACE_ground_truth.csv'),
                        help='Path to the .csv file containing the ground truth for the dataset')
    parser.add_argument('-dbp', dest='dataset_partition_file', type=str, required=True,
                        help="Path to the .csv file containing dataset partitioning into 'train', 'valid' and "
                             "'test' sets")
    parser.add_argument('-extractor_id', dest='extractor_id', type=str, default='COLOR',
                        help="Data to be extracted. Currently only 'COLOR' is supported.")
    parser.add_argument('-bs', dest='batch_size', type=int, default=16,
                        help="Batch size to be used when creating the loaders.")

    print('Done parsing ... ')
    args = parser.parse_args()

    # Load pre-processed data
    data_dict = load_preprocessed_data(args.data_path)

    # Create dataset based on provided extractor_id and provided partition
    dataset = create_padisi_face_dataset(extractor_id=args.extractor_id,
                                         db_path=None,  # Used only when raw data is available.
                                         gt_path=args.ground_truth_file,
                                         part_path=args.dataset_partition_file,
                                         preprocessed_data_dict=data_dict)

    # Create loader for each partition - store them in a dictionary
    partition_names = ('train', 'valid', 'test')
    partitions = dict()
    loaders = dict()
    for i, partition_name in enumerate(partition_names):
        partitions[partition_name] = dataset.get_partition(partition_name)
        # Convert partition to a PyTorch loader, if PyTorch available
        if torch is not None:
            loaders[partition_name] = convert_partition_to_loader(partition=partitions[partition_name],
                                                                  partition_name=partition_name,
                                                                  batch_size=args.batch_size)
        else:
            if i == 0:
                print('\n\nNo PyTorch installation found. Will not create data loader and just loop trough the data.')

    # No PyTorch available - just looping through the data
    if len(loaders) == 0:
        for partition_name, partition in partitions.items():
            print("Checking '{}' partition.".format(partition_name))
            if len(partition) == 0:
                print("-- No '{}' partition found in data. Skipping...\n".format(partition_name))
                continue
            for sample_id, reader in enumerate(partition):
                print("Checking sample {}/{}".format(sample_id + 1, len(partition)))
                print('  -- Sample info       : {}'.format(reader.get_reader_info_str()))
                print('  -- Sample identifier : {}'.format(reader.get_reader_identifier()))
                print('  -- Sample shape      : {}'.format(reader.get_data().shape))
                print('  -- Sample label      : {}'.format(reader.get_ground_truth()))

    else:
        # Loop through each loader to verify correct creation
        for partition_name, loader in loaders.items():
            print("Checking '{}' loader".format(partition_name))
            if loader is None:
                print("-- No '{}' partition found in data. Skipping...\n".format(partition_name))
                continue

            for batch_id, (data, labels) in enumerate(loader):
                print('Reading batch {}/{}'.format(batch_id + 1, len(loader)))
                print('  -- Batch data shape   : {}'.format(data.shape))
                print('  -- Batch labels shape : {}'.format(labels.shape))

    print('\nChecking finished successfully!')


if __name__ == '__main__':
    main()
