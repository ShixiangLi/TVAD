# convert_mat_to_pt.py
import os
import h5py
import numpy as np
import torch

def get_mat_files_from_directory(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.mat')]

def convert_and_save(directory, output_path='fmf_dataset.pt'):
    current_data, labels, train_test_index, video_data = [], [], [], []

    mat_files = get_mat_files_from_directory(directory)
    for mat_file in mat_files:
        with h5py.File(mat_file, 'r') as f:
            current_data.append(np.array(f['current']))
            labels.append(np.array(f['label']))
            train_test_index.append(np.array(f['train_test_index']).flatten())
            video_data.append(np.array(f['video']))

    data_dict = {
        'current': np.concatenate(current_data, axis=0),
        'label': np.concatenate(labels, axis=0),
        'train_test_index': np.concatenate(train_test_index, axis=0),
        'video': np.concatenate(video_data, axis=0),
    }

    torch.save(data_dict, output_path)
    print(f"Saved preprocessed data to {output_path}")

if __name__ == "__main__":
    data_dir = 'data/FMFBenchmarkV1'  # 改成你的路径
    convert_and_save(data_dir)
