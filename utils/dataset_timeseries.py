import numpy as np
import pandas as pd
import os
from utils.utils import to_categorical
import torch.utils.data as data
import torch

def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a

def load_UCR_data(root, file_name='', normalize_timeseries=2, verbose=True):

    train_name = '_'.join([file_name, 'TRAIN'])
    test_name = '_'.join([file_name, 'TEST'])
    data_path = os.path.join(root, file_name)

    df = pd.read_csv(os.path.join(data_path, train_name), header=None, encoding='latin-1')

    is_timeseries = True # assume all input data is univariate time series

    # remove all columns which are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    if not is_timeseries:
        data_idx = df.columns[1:]
        min_val = min(df.loc[:, data_idx].min())
        if min_val == 0:
            df.loc[:, data_idx] += 1

    # fill all missing columns with 0
    df.fillna(0, inplace=True)

    # cast all data into integer (int32)
    if not is_timeseries:
        df[df.columns] = df[df.columns].astype(np.int32)

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    y_train = df[[0]].values
    nb_classes = len(np.unique(y_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    y_train = np.squeeze(y_train)

    # drop labels column from train set X
    df.drop(df.columns[0], axis=1, inplace=True)

    X_train = df.values

    if is_timeseries:
        X_train = X_train[:, np.newaxis, :]
        # scale the values
        if normalize_timeseries:
            normalize_timeseries = int(normalize_timeseries)

            if normalize_timeseries == 2:
                X_train_mean = X_train.mean()
                X_train_std = X_train.std()
                X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)

            else:
                X_train_mean = X_train.mean(axis=-1, keepdims=True)
                X_train_std = X_train.std(axis=-1, keepdims=True)
                X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)

    if verbose: print("Finished loading train dataset..")

    df = pd.read_csv(os.path.join(data_path, test_name), header=None, encoding='latin-1')

    # remove all columns which are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    if not is_timeseries:
        data_idx = df.columns[1:]
        min_val = min(df.loc[:, data_idx].min())
        if min_val == 0:
            df.loc[:, data_idx] += 1

    # fill all missing columns with 0
    df.fillna(0, inplace=True)

    # cast all data into integer (int32)
    if not is_timeseries:
        df[df.columns] = df[df.columns].astype(np.int32)

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    y_test = df[[0]].values
    nb_classes = len(np.unique(y_test))
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)
    y_test = np.squeeze(y_test)

    # remove all columns which are completely empty
    df.dropna(axis=1, how='all', inplace=True)
    # fill all missing columns with 0
    df.fillna(0, inplace=True)

    # drop labels column from train set X
    df.drop(df.columns[0], axis=1, inplace=True)

    X_test = df.values

    if is_timeseries:
        X_test = X_test[:, np.newaxis, :]
        # scale the values
        if normalize_timeseries:
            normalize_timeseries = int(normalize_timeseries)

            if normalize_timeseries == 2:
                X_test = (X_test - X_train_mean) / (X_train_std + 1e-8)
            else:
                X_test_mean = X_test.mean(axis=-1, keepdims=True)
                X_test_std = X_test.std(axis=-1, keepdims=True)
                X_test = (X_test - X_test_mean) / (X_test_std + 1e-8)

    if verbose:
        print("Finished loading test dataset..")
        print()
        print("Number of train samples : ", X_train.shape[0], "Number of test samples : ", X_test.shape[0])
        print("Number of classes : ", nb_classes)
        print("Sequence length : ", X_train.shape[-1])

    return X_train, y_train, X_test, y_test, nb_classes


def load_dataset_mul(dataset_path, dataset_name, normalize_timeseries=False, verbose=True) -> (np.array, np.array):
    if verbose: print("Loading train / test dataset : ", dataset_name)

    root_path = dataset_path + '/' + dataset_name + '/'
    x_train_path = root_path + "X_train.npy"
    y_train_path = root_path + "y_train.npy"
    x_test_path = root_path + "X_test.npy"
    y_test_path = root_path + "y_test.npy"

    if os.path.exists(x_train_path):
        X_train = np.load(x_train_path).astype(np.float32)
        y_train = np.squeeze(np.load(y_train_path))
        X_test = np.load(x_test_path).astype(np.float32)
        y_test = np.squeeze(np.load(y_test_path))
    else:
        raise FileNotFoundError('File %s not found!' % (dataset_name))

    is_timeseries = True

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    nb_classes = len(np.unique(y_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)

    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            X_train_mean = X_train.mean()
            X_train_std = X_train.std()
            X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)

    if verbose: print("Finished processing train dataset..")

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    nb_classes = len(np.unique(y_test))
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            X_test = (X_test - X_train_mean) / (X_train_std + 1e-8)

    if verbose:
        print("Finished loading test dataset..")
        print()
        print("Number of train samples : ", X_train.shape[0], "Number of test samples : ", X_test.shape[0])
        print("Number of classes : ", nb_classes)
        print("Sequence length : ", X_train.shape[-1])

    return X_train, y_train, X_test, y_test, nb_classes

def TSC_data_loader_128(dataset_path, dataset_name, normalize_timeseries=2):
    Train_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.tsv')
    Test_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TEST.tsv')

    X_train = Train_dataset[:, 1:].astype(np.float32)
    y_train = Train_dataset[:, 0:1]

    X_test = Test_dataset[:, 1:].astype(np.float32)
    y_test = Test_dataset[:, 0:1]

    X_train = set_nan_to_zero(X_train)
    X_test = set_nan_to_zero(X_test)

    is_timeseries = True

    if is_timeseries:
        X_train = X_train[:, np.newaxis, :]
        # scale the values
        if normalize_timeseries:
            normalize_timeseries = int(normalize_timeseries)

            if normalize_timeseries == 2:
                X_train_mean = X_train.mean()
                X_train_std = X_train.std()
                X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)

            else:
                X_train_mean = X_train.mean(axis=-1, keepdims=True)
                X_train_std = X_train.std(axis=-1, keepdims=True)
                X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)

    if is_timeseries:
        X_test = X_test[:, np.newaxis, :]
        # scale the values
        if normalize_timeseries:
            normalize_timeseries = int(normalize_timeseries)

            if normalize_timeseries == 2:
                X_test = (X_test - X_train_mean) / (X_train_std + 1e-8)
            else:
                X_test_mean = X_test.mean(axis=-1, keepdims=True)
                X_test_std = X_test.std(axis=-1, keepdims=True)
                X_test = (X_test - X_test_mean) / (X_test_std + 1e-8)


    # X_train = X_train[:, np.newaxis, :]
    # X_test = X_test[:, np.newaxis, :]

    nb_classes = len(np.unique(y_test))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    print("Finished loading test dataset..")
    print()
    print("Number of train samples : ", X_train.shape[0], "Number of test samples : ", X_test.shape[0])
    print("Number of classes : ", nb_classes)
    print("Sequence length : ", X_train.shape[-1])

    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test, nb_classes


class Data(data.Dataset):

    def __init__(self, train=True, x_train=None, y_train=None, x_test=None, y_test=None):
        self.train_data = x_train
        self.train_labels = y_train
        self.test_data = x_test
        self.test_labels = y_test
        self.train = train

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

def get_timeseries_dataset(batch_size=32, x_train=None, y_train=None, x_test=None, y_test=None, n_worker=1):
    # n_classes = len(np.unique(y_train))
    # y_train = to_categorical(y_train, n_classes)
    # y_test = to_categorical(y_test, n_classes)

    trainset = Data(train=True, x_train=x_train, y_train=y_train, x_test=None, y_test=None)
    valset = Data(train=False, x_train=None, y_train=None, x_test=x_test, y_test=y_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                               num_workers=n_worker, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False,
                                             num_workers=n_worker, pin_memory=True)

    return train_loader, val_loader


