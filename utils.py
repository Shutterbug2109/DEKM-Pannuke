"""
This module contains functions for data preprocessing and metrics calculation for Deep Embedded K Means.

Paper : https://arxiv.org/pdf/2109.15149.pdf#:~:text=RED%2D%20KC%20(for%20Robust%20Embedded,representation%20learning%20and%20clustering.

Github : https://github.com/spdj2271/DEKM .

"""
import csv
import os
import os.path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import normalized_mutual_info_score
from sklearn import metrics
#from scipy.optimize import linear_sum_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment

# pylint: disable=C0103

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def get_metrics(_y, _y_pred):
    """
    Function to calculate metrics like  ACC: Unsupervised clusterng accuracy, NMI: normalized mutual info,
    FMS : fowlkes_mallows_score, hcv : homogeneity_completeness_v_measure, mis : mutual info score, rs : rand score
    Args: 
        y = lables
        y_pred = predicted labels
    Returns : 
        acc, nmi,fms,hcv,mis,rs.
    """
    # y = np.array(_y)
    # y_pred = np.array(_y_pred)
    # s = np.unique(y_pred)
    # # print(s)
    # t = np.unique(y)
    # # print(t)

    # C = np.zeros((N, N), dtype=np.int32)
    # for i in range(N):
    #     for j in range(N):
    #         idx = np.logical_and(y_pred == s[i], y == t[j])
    #         C[i][j] = np.count_nonzero(idx)
    # Cmax = np.amax(C)
    # C = Cmax - C 

    # row, col = linear_sum_assignment(C)
    # count = 0
    # for i in range(N):
    #     idx = np.logical_and(y_pred == s[row[i]], y == t[col[i]])
    #     count += np.count_nonzero(idx)
    # acc = np.round(1.0 * count / len(y), 5)

    y_pred, y = np.array(_y_pred).astype(int), np.array(_y).astype(int)
    assert y_pred.size == y.size
    D = max(y_pred.max(), y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y[i]] += 1
    row, col = linear_assignment(w.max() - w)

    acc = sum([w[row[i],col[i]] for i in range(row.shape[0])]) * 1.0 / y_pred.size

    N = len(np.unique(y_pred))

    temp = np.array(y_pred)
    for i in range(N):
        y_pred[temp == col[i]] = i
    # normalized mutual info
    nmi = np.round(normalized_mutual_info_score(y, y_pred), 5)
    # fowlkes_mallows_score
    fms = metrics.fowlkes_mallows_score(y, y_pred)
    # homogeneity_completeness_v_measure
    hcv = metrics.homogeneity_completeness_v_measure(y, y_pred)
    mis = metrics.mutual_info_score(y, y_pred)  # mutual info score
    rs = metrics.rand_score(y, y_pred)  # rand score

    return acc, nmi, fms, hcv, mis, rs


def get_xy(
    ds_name="PANNUKE",
    dir_path="../data/",
    log_print=True,
    shuffle_seed=None,
    gdrive=False,
):
    """
    Function to get x and y from the dataset, where x is : images, y is : labels.
    Args:
        ds_name= Datasets,
        dir_path= path to data folder,
        log_print= True,
        shuffle_seed= None,
        gdrive= False

    Returns : 
        x = Image arrays
        y = Labels

    """
    if ds_name == "PANNUKE_DILATED":
        if gdrive:
            directory = (
                "/content/drive/MyDrive/Colab Notebooks/Augmented_Dilated_PANNUKE/"
            )
        else:
            directory = dir_path + "Augmented_Dilated_PANNUKE/"
        # set the data path and corresponding csv file in the same folder
        df = pd.read_csv(directory + "PANNUKE_Dilated_Augmented.csv")
    elif ds_name == "PANNUKE":
        # set the data path and corresponding csv file in the same folder
        if gdrive:
            directory = (
                "/content/drive/MyDrive/Colab Notebooks/Augmented_PanNuke/"
            )
        else:
            directory = dir_path + "Augmented_PanNuke/"
        df = pd.read_csv(directory + "PANNUKE_Augmented.csv")

    elif ds_name == "PANNUKE_ONLYCELLS":
        # set the data path and corresponding csv file in the same folder
        if gdrive:
            directory = (
                "/content/drive/MyDrive/Colab Notebooks/OnlyCell_Augmented_PanNuke/"
            )
        else:
            directory = dir_path + "OnlyCell_Augmented_PanNuke/"
        df = pd.read_csv(directory + "OnlyCellPanNuke.csv")

    file_paths = df["FileName"].values
    labels = df["Labels"].values
    ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    def read_image(image_file, label):
        image = tf.io.read_file(directory + image_file)
        image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
        return image / 255.0, label

    # getting all the images which will the divided into batches as per the batch size
    ds_train = ds_train.map(read_image).batch(10487)  
    for image, labels in ds_train:
        x = image
        y = labels

    if not shuffle_seed:
        shuffle_seed = int(np.random.randint(100))
    idx = np.arange(0, len(x))
    idx = tf.random.shuffle(idx, seed=shuffle_seed).numpy()
    # x = x[idx]
    # y = y[idx]
    x = tf.random.shuffle(x, seed=shuffle_seed).numpy()  # converting to numpy
    y = tf.random.shuffle(y, seed=shuffle_seed).numpy()  # converting to numpy
    if log_print:
        print(ds_name)
    return x, y


def log_csv(strToWrite, file_name):
    """Function to log the the metrics into csv file"""
    path = r"log_history/"
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(path + file_name + ".csv", "a+", encoding="utf-8")
    csv_writer = csv.writer(f)
    csv_writer.writerow(strToWrite)
    f.close()


def read_list(file_name, type):
    """Function read_list reads the file and remove the spaces if any."""
    with open(file_name, encoding="utf-8") as f:
        lines = f.readlines()
    if type == "str":
        array = np.asarray([l.strip() for l in lines])
        return array
    elif type == "int":
        array = np.asarray([int(l.strip()) for l in lines])
        return array
    else:
        print("Unknown type")
        return None
