#These are common utilities that are used by many different scripts

import numpy as np
import torch
import os
import glob

#Our package imports
import models


def get_cnn_models(model_dir = 'models'):
    '''Returns a list of the CNN model names and number of training points

    Parameters
    ----------
    model_dir : (optional, default = 'models')

    Returns
    -------
    cnn_models : A python list of the strings containing all CNN models
                 in model_dir ending in *.pt
    cnn_train_idx : A pytyhon list of .npy files giving training points for each model
    cnn_num_train : A python list giving the number of training points for each model
    '''


    #Retrieve CNN model names and number of training points
    cnn_models = glob.glob('models/MNIST_CNN_*.pt')
    cnn_num_train = [int(f[20:-3]) for f in cnn_models]

    #Sort models by number of training points
    I = np.argsort(cnn_num_train)
    cnn_num_train = [cnn_num_train[i] for i in I]
    cnn_models = [cnn_models[i] for i in I]
    cnn_train_idx = [cnn_models[i][:-3]+'_training_indices.npy' for i in range(len(I))]

    return cnn_models, cnn_train_idx, cnn_num_train

def NormalizeData(data):
    '''Normalizes data to range [0,1]

    Parameters
    ----------
    data : Numpy array

    Returns
    -------
    norm_data : Normalized array
    '''

    norm_data = (data - np.min(data))/(np.max(data) - np.min(data))
    return norm_data


def encodeMNIST(data, model_path, batch_size = 1000, cuda = True, use_phase = False):
    '''Load a torch CNN model and encode MNIST

    Parameters
    ----------
    model_path : Path to .pt file containing torch model for trained CNN
    batch_size : Size of minibatches to use in encoding. Reduce if you get out of memory errors (default = 1000)
    cuda : Whether to use GPU or not (default = True)
    use_phase : Whether the model uses phase information or not (default = False)

    Returns
    -------
    encoded_data : Returns a numpy array of MSTAR encoded by model.encode() (e.g., the CNN features)
    '''

    use_cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    data = torch.from_numpy(data).float()

    #Load model
    model = torch.load(model_path, map_location=device)
    model.eval()
    encoded_data = None
    with torch.no_grad():
        for idx in range(0,len(data),batch_size):
            data_batch = data[idx:idx+batch_size]
            if encoded_data is None:
                encoded_data = model.encode(data_batch.to(device)).cpu().numpy()
            else:
                encoded_data = np.vstack((encoded_data,model.encode(data_batch.to(device)).cpu().numpy()))

    return encoded_data
