# Note that the IRMAS training dataset was used. A smaller subset of the entire dataset was extract and moved to a zip file
# this module needs a revision for future use
# import all libraries we need here before starting 

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim 
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.nn as nn
import torchaudio
import numpy as np
import math
import os 
import shutil
use_cuda = True


def normalize_waveform(waveform, norm_max=0.5):
    max_magnitudes = waveform.abs().max(dim=1, keepdim=True)[0]
    normalized_waveforms = waveform.float().div(max_magnitudes) * norm_max
    return normalized_waveforms

def audio_loader(file_path):
    waveform, _ = torchaudio.load(file_path)
    return normalize_waveform(waveform, Normalized_Max)

def get_data_indices(data_size):
    # Randomly split data into training, validation and test sets.

    # Create a list of randomized indices of image data
    np.random.seed(1)
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    # Set size for each dataset
    validation_size = math.floor(data_size * VALIDATION_PERCENTAGE)
    test_size = math.floor(data_size * TEST_PERCENTAGE)
    training_size = data_size - validation_size - test_size

    training_indices = indices[:training_size]
    val_indices = indices[training_size : training_size + validation_size]
    test_indices = indices[training_size + validation_size:]

    return training_indices, val_indices, test_indices

def get_data_loaders(folder, batch_size=64): 
    # Load training, validation and test data.

    data_size = len(folder)

    # Get training, validation and test data indices
    training_indices, val_indices, test_indices = get_data_indices(data_size)

    # Create subsets
    training_set = torch.utils.data.Subset(folder, training_indices)
    val_set = torch.utils.data.Subset(folder, val_indices)
    test_set = torch.utils.data.Subset(folder, test_indices)

    # Create dataloaders for each dataset
    train_loader = Data.DataLoader(training_set, batch_size=batch_size)
    validation_loader = Data.DataLoader(val_set, batch_size=batch_size)
    test_loader = Data.DataLoader(test_set, batch_size=batch_size)
    return train_loader, validation_loader, test_loader

def get_one_hot_targets(class_index):
    return one_hot_class_targets[class_index]

def audio_loader(file_path):
    waveform, _ = torchaudio.load(file_path)
    return waveform

def combine_data():

    # needs revisioning

    input_dir_path = "/root/IRMAS-Small" 
    combined_dir_path = "/root/IRMAS-Combine"

    # Make a directory in /root/
    os.mkdir(combined_dir_path)

    class_directories = os.listdir(input_dir_path)
    file_count = len(os.listdir(os.path.join(input_dir_path, class_directories[0])))

    # create files with combined and normalized audio
    for i in range(len(class_directories)):
    for j in range(i+1, len(class_directories)):
        dir_name = class_directories[i] + '+' + class_directories[j]
        dir_path = os.path.join(combined_dir_path, dir_name)
        if os.path.isdir(dir_path):
        for file_name_to_remove in os.listdir(dir_path):
            os.remove(os.path.join(dir_path, file_name_to_remove))
        os.rmdir(dir_path)
        os.mkdir(dir_path)
        class1_names = os.listdir(os.path.join(input_dir_path, class_directories[i]))
        class2_names = os.listdir(os.path.join(input_dir_path, class_directories[j]))
        for file_index in range(file_count):
        wave1, sample_rate = torchaudio.load(os.path.join(input_dir_path, class_directories[i], class1_names[file_index]))
        wave2, sample_rate = torchaudio.load(os.path.join(input_dir_path, class_directories[j], class2_names[file_index]))
        wave1 = normalize_waveform(wave1, 0.5)
        wave2 = normalize_waveform(wave2, 0.5)
        combined_wave = wave1 + wave2
        combined_wave = normalize_waveform(combined_wave, 0.5)
        file_name = dir_name + str(file_index) + ".wav"
        torchaudio.save(os.path.join(dir_path, file_name), combined_wave, sample_rate=sample_rate)

    target_transform = transforms.Compose([
                                transforms.Lambda(get_one_hot_targets)
                               ])

    combined_audio_folder = torchvision.datasets.DatasetFolder("/root/IRMAS-Combine", loader=audio_loader, target_transform=target_transform, extensions='wav')
    return combined_audio_folder
  

if __name__ == '__main__':
    # unzip the small dataset zipfile using cmd on Linux OS:
    # !unzip '/content/drive/My Drive/APS 360 Project/IRMAS-Training-Small.zip' -d '/root/'
    # On Windows, unzip the file normally using 'Extract all'
    
    # Stage 1:

    VALIDATION_PERCENTAGE = 0.2
    TEST_PERCENTAGE = 0.2
    classes = ['gac','pia','tru','vio']
    train_dir = "/root/IRMAS-Small"
    for file in os.listdir(train_dir):
    if file not in classes:
        os.remove(os.path.join(train_dir, file))
        print("Removed {}".format(file))

    Normalized_Max = 0.5
    audioFolder = torchvision.datasets.DatasetFolder("/root/IRMAS-Small", loader=audio_loader, extensions='wav')

    train_loader, val_loader, test_loader = get_data_loaders(audioFolder, 1)

    # Output the size of each dataset.
    print("# of training examples: ", len(train_loader))
    print("# of validation examples: ", len(val_loader))
    print("# of test examples: ", len(test_loader))

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        print(inputs[0].shape)

        break
    
    # *********************************************************************************************
    # Stage 2:

    # Combine audio files
    combined_audio = combine_data()

    