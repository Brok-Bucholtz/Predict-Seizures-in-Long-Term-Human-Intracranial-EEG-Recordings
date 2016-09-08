from glob import glob

from os import path
from scipy.io import loadmat
from tqdm import tqdm

from helper import matfile_metadata, METADATA_COLUMNS, filename_to_preictal


def print_unique_metadata():
    metadata = {column: [] for column in METADATA_COLUMNS}
    file_paths = glob('./input/*/*.mat')
    with tqdm(total=len(file_paths), unit='File') as progress_bar:
        for file_path in file_paths:
            progress_bar.update()
            if path.isfile(file_path):
                try:
                    mat_file = loadmat(file_path)['dataStruct']
                except ValueError as err:
                    print('Error in file \"{}\": \"{}\"'.format(file_path, err))
                else:
                    file_metadata = matfile_metadata(mat_file)

                    # Check if training data, before attempting to get label from filename
                    if file_path.startswith('./input/train'):
                        file_metadata['is_preictal'] = [filename_to_preictal(file_path)]

                    for column in file_metadata.keys():
                        metadata_value = file_metadata[column]
                        if len(metadata_value) == 1:
                            metadata_value = metadata_value[0]
                        else:
                            metadata_value = ', '.join([str(value) for value in metadata_value])
                        metadata[column].append(metadata_value)
    counts = [(column, [(value, metadata[column].count(value)) for value in set(metadata[column])])
              for column in metadata.keys()]
    for column, set_counts in counts:
        print('\n{} has {} unique value:'.format(column, len(set_counts)))
        for value, count in set_counts:
            print('   \"{}\" -  {}'.format(value, count))


if __name__ == '__main__':
    print('Getting unique metadata values...')
    print_unique_metadata()
