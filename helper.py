from glob import glob
from os import path

from scipy.io import loadmat

import pandas as pd


def mat_to_dataframe(files):
    main_data_size = 16
    meta_data_columns = ['iEEGsamplingRate', 'nSamplesSegment', 'channelIndices', 'sequence', 'is_preictal']
    all_data = {}

    # Setup dictionary to store data
    for meta_column in meta_data_columns:
        all_data['meta', meta_column] = {}
    for data_i in range(main_data_size):
        all_data['data', data_i] = {}

    for file_path in files:
        if path.isfile(file_path):
            file = path.splitext(path.basename(file_path))[0]
            patient_id, segment_i, is_preictal = file.split('_')
            patient_id = int(patient_id)
            segment_i = int(segment_i)
            is_preictal = is_preictal == '1'
            data_struct = loadmat(file_path)['dataStruct']

            # Add mat data to dictionary
            for column in meta_data_columns:
                if column == 'is_preictal':
                    all_data['meta', 'is_preictal'][patient_id, segment_i] = is_preictal
                else:
                    all_data['meta', column][patient_id, segment_i] = data_struct[column][0, 0][0, 0]
            for data_i, data in enumerate(data_struct['data'][0, 0].transpose()):
                all_data['data', data_i][patient_id, segment_i] = data
    return pd.DataFrame(all_data)


def all_training_files():
    return glob('./input/train_*/*.mat')


def all_test_files():
    return glob('./input/test_*/*.mat')
