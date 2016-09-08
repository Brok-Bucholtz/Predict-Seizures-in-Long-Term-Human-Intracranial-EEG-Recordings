from glob import glob
from os import path

from scipy.io import loadmat

import pandas as pd


METADATA_COLUMNS = ['iEEGsamplingRate', 'nSamplesSegment', 'channelIndices', 'sequence', 'is_preictal']


def filename_metadata(file_path):
    file = path.splitext(path.basename(file_path))[0]
    patient_id, segment_i, is_preictal = file.split('_')
    patient_id = int(patient_id)
    segment_i = int(segment_i)
    is_preictal = is_preictal == '1'

    return {'patient_id': patient_id, 'segment_i': segment_i, 'is_preictal': is_preictal}


def matfile_metadata(matfile):
    return {column: matfile[column][0, 0][0, 0] for column in METADATA_COLUMNS if column != 'is_preictal'}


def mat_to_dataframe(files):
    main_data_size = 16
    all_data = {}

    # Setup dictionary to store data
    for meta_column in METADATA_COLUMNS:
        all_data['meta', meta_column] = {}
    for data_i in range(main_data_size):
        all_data['data', data_i] = {}

    for file_path in files:
        if path.isfile(file_path):
            try:
                mat_file = loadmat(file_path)['dataStruct']
            except ValueError as err:
                print('Error in file \"{}\": \"{}\"'.format(file_path, err))
            else:
                metadata = {**filename_metadata(file_path), **matfile_metadata(mat_file)}
                index = (metadata['patient_id'], metadata['segment_i'])

                # Add mat data to dictionary
                for column in METADATA_COLUMNS:
                    all_data['meta', column][index] = metadata[column]
                for data_i, data in enumerate(mat_file['data'][0, 0].transpose()):
                    all_data['data', data_i][index] = data
    return pd.DataFrame(all_data)


def all_training_files():
    return glob('./input/train_*/*.mat')


def all_test_files():
    return glob('./input/test_*/*.mat')
