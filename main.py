"""
This module is the main Entrypoint .

Paper : https://arxiv.org/pdf/2109.15149.pdf#:~:text=RED%2D%20KC%20(for%20Robust%20Embedded,representation%20learning%20and%20clustering.

Github : https://github.com/spdj2271/DEKM .
"""


import argparse
import time
import tensorflow as tf
from functions import Implementation_DEKM
from utils import get_xy

# pylint: disable=C0103

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="select dataset PANNUKE,PANNUKE_ONLYCELLS,PANNUKE_DILATED"
    )
    parser.add_argument("--ds_name", default="PANNUKE")
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--pretrain_epochs", type=int, default=2)
    parser.add_argument("--hidden_units", type=int, default=1)
    parser.add_argument("--environment", default="GDRIVE")
    args = parser.parse_args()

    params = {
        "pretrain_epochs": args.pretrain_epochs,
        "pretrain_batch_size": 256,
        "batch_size": 256,
        "update_interval": 40,
        "hidden_units": args.hidden_units,
        "n_clusters": args.n_clusters,
    }
    if args.ds_name is None or not args.ds_name in [
        "PANNUKE",
        "PANNUKE_ONLYCELLS",
        "PANNUKE_DILATED",
    ]:
        ds_name = "PANNUKE"
    else:
        ds_name = args.ds_name

    if ds_name == "PANNUKE":
        input_shape = (56, 56, 3)
        n_clusters = 4
    elif ds_name == "PANNUKE_ONLYCELLS":
        input_shape = (56, 56, 3)
        n_clusters = 4
    elif ds_name == "PANNUKE_DILATED":
        input_shape = (64, 64, 3)
        n_clusters = 4

    time_start = time.time()
    if args.environment == "GDRIVE":
        x, y = get_xy(ds_name=ds_name)
    else:
        x, y = get_xy(ds_name=ds_name, gdrive=False)

    ds_xx = (
        tf.data.Dataset.from_tensor_slices((x, x))
        .shuffle(8000)
        .batch(params["pretrain_batch_size"])
    )
    DeepKMeans = Implementation_DEKM(
        input_shape=input_shape,
        ds_name=ds_name,
        hidden_units=args.hidden_units,
        pretrain_epochs=args.pretrain_epochs,
        n_clusters=args.n_clusters,
    )
    DeepKMeans.train_base(ds_xx)
    DeepKMeans.train(x, y, params)
    # stop neptune logging ( must in colab )
    print(time.time() - time_start)
