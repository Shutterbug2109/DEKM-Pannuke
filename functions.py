"""
This module defines a class to implement the the Deep Embedded K Means.

Paper : https://arxiv.org/pdf/2109.15149.pdf#:~:text=RED%2D%20KC%20(for%20Robust%20Embedded,representation%20learning%20and%20clustering.

Github : https://github.com/spdj2271/DEKM .

"""

import time
import numpy as np
import tensorflow as tf
import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from tensorflow import keras
from sklearn.cluster import KMeans
from keras import layers
from keras import losses
from keras.models import Model
from utils import get_metrics
from utils import log_csv
from scipy.optimize import linear_sum_assignment as linear_assignment

# pylint: disable=C0103


# Neptune API
run = neptune.init(
    project="nikita.deshpande/Deep-Embedded-KMeans",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1MjhlNDc0Zi03NzAwLTRjZjktOGYyZi05OWY1MGQyZDA1ZDAifQ==",
)  # your credentials

# Neptune Callback
neptune_cbk = NeptuneCallback(run=run, base_namespace="training")

# Neptune log metrics during training
class NeptuneLogger(keras.callbacks.Callback):
    """
    Docstring
    """

    def on_batch_end(self, batch, logs=None):       
        if logs is None:
            logs = {}
        for log_name, log_value in logs.items():
            run["batch/{}".format(log_name)].log(log_value)

    def on_epoch_end(self, epoch, logs=None):     
        if logs is None:
            logs = {}
        for log_name, log_value in logs.items():
            run["epoch/{}".format(log_name)].log(log_value)


class Implementation_DEKM:
    """
    This class depicts the implementation of Deep Embedded K Means.
    """
    def __init__(self, input_shape, ds_name, hidden_units, pretrain_epochs, n_clusters):
        """
        Args :
            input shape = Shape of the input images e.g (56,56,3)
            ds_name(str)= Name of the dataset
            hidden_units(int) = hidden_units
            pretrain_epochs(int) = Number of epochs
            n_clusters(int) = Number of clusters. 
        """ 
        self.input_shape = input_shape
        self.ds_name = ds_name
        self.hidden_units = hidden_units
        self.pretrain_epochs = pretrain_epochs
        self.n_clusters = n_clusters

    def model_conv(self, load_weights=True):
        """ 
        This method is a Implementation of Convolutional Autoencoder model. 

        Args : 
            load_weight = The value is set to True

        Discription :
        - It use three convolutional layers followed by a dense layer (embedding layer) in the
            encoder-to-decoder pathways.
        - The channel numbers of the three.
            convolutional layers are 32, 64, and 128 respectively. 
        - The kernel sizes are set to 5×5, 5×5, and 3×3 respectively.
        - The stride of all the convolutional layers is set to two.
        - The number of neurons in the embedding layer is set to the number of clusters of datasets.
        - The decoder is a mirror of the encoder and the output of each layer of the decoder
            is appropriately zero-padded to match the input size of the corresponding encoder layer
        - All the intermediate layers of the convolutional autoencoder are activated by ReLU

        returns : model
        """
        # init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
        init = "uniform"
        filters = [32, 64, 128, self.hidden_units]
        if self.input_shape[0] % 8 == 0:
            pad3 = "same"
        else:
            pad3 = "valid"
        input = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(
            filters[0],
            kernel_size=5,
            strides=2,
            padding="same",
            activation="relu",
            kernel_initializer=init,
        )(input)
        x = layers.Conv2D(
            filters[1],
            kernel_size=5,
            strides=2,
            padding="same",
            activation="relu",
            kernel_initializer=init,
        )(x)
        x = layers.Conv2D(
            filters[2],
            kernel_size=3,
            strides=2,
            padding=pad3,
            activation="relu",
            kernel_initializer=init,
        )(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=filters[-1], name="embed")(x)
        #     x = tf.divide(x, tf.expand_dims(tf.norm(x, 2, -1), -1))
        h = x
        x = layers.Dense(
            filters[2] * (self.input_shape[0] // 8) * (self.input_shape[0] // 8),
            activation="relu",
        )(x)
        x = layers.Reshape(
            (self.input_shape[0] // 8, self.input_shape[0] // 8, filters[2])
        )(x)
        x = layers.Conv2DTranspose(
            filters[1], kernel_size=3, strides=2, padding=pad3, activation="relu"
        )(x)
        x = layers.Conv2DTranspose(
            filters[0], kernel_size=5, strides=2, padding="same", activation="relu"
        )(x)
        x = layers.Conv2DTranspose(
            self.input_shape[2], kernel_size=5, strides=2, padding="same"
        )(x)
        output = layers.Concatenate()([h, layers.Flatten()(x)])
        model = Model(inputs=input, outputs=output)
        model.summary()
        if load_weights:
            model.load_weights(f"log_history/weight_base_{self.ds_name}.h5")
            print("model_conv: weights was loaded")

        return model

    def loss_train_base(self, y_true, y_pred):
        """
        This function calculates the loss of Autoencoder
        Args:
            y_true = True labels
            y_pred = Predicted labels
        Discription:
            Loss of AE : Mean squared Error.
        """
        y_true = layers.Flatten()(y_true)
        y_pred = y_pred[:, self.hidden_units:]
        return losses.mse(y_true, y_pred)

    def train_base(self, ds_xx):
        """
        This function is to train the base model (convolutional AE)
        For AE we give same input and output to the network eg (x,x)
        instead of (x,y).
        Args: 
            ds_xx = tf.data.Dataset.from_tensor_slices((x, x)).
        Discription: 
        - The Adam optimizer is adopted with the initial learning rate l = 0.001, β1 = 0.9,
        β2 = 0.999. 
        - The clustering process is stopped when there are less
        than 0.1% of samples that change their clustering assignments
        between two consecutive iterations.
        """
        model = self.model_conv(load_weights=False)
        model.compile(optimizer="adam", loss=self.loss_train_base)
        # neptune_cbk = NeptuneCallback(run=run, base_namespace="training")
        model.fit(
            ds_xx, epochs=self.pretrain_epochs, verbose=2, callbacks=[NeptuneLogger()]
        )
        model.save_weights(f"log_history/weight_base_{self.ds_name}.h5")
        print("This is a base model")

    def sorted_eig(self, X):
        """Function contains the eigenvectors sorted in ascending order w.r.t. their eigenvalues.
        Args:
            X = within-class scatter matrix
        Returns : 
            e_vecs, e_vals
            
        """
        e_vals, e_vecs = np.linalg.eig(X)  # The feature vector v[:,i] corresponds to the feature value w[i], that is, each feature vector in each column
        idx = np.argsort(e_vals)
        e_vecs = e_vecs[:, idx]
        e_vals = e_vals[idx]
        return e_vals, e_vecs

    def train(self, x, y, params, time_start=time.time()):
        """ Implemet the K Means clustering and also calculate the metrics
        Args:
            x = Image data
            y = Corresponding labels
            parms = params = {
                            "pretrain_epochs": args.pretrain_epochs,
                            "pretrain_batch_size": 256,
                            "batch_size": 256,
                            "update_interval": 40,
                            "hidden_units": args.hidden_units,
                            "n_clusters": args.n_clusters,
                            }
            time_start = time

        Discription : 
            We can first perform K-means in the embedding space H to get Sw, and then eigendecompose
            Sw to get V. Finally, transform the embedding space to a new space Y that reveals the cluster-structure information. 
            Also know the importance of each dimension of Y in terms of
            the cluster-structure information, i.e., the last dimension has
            the least cluster-structure information.

        """
        run["hyper-parameters"] = params
        log_str = f'iter; acc, nmi, ri, fms, hcv,mis, rs ; loss; n_changed_assignment;time:{time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}'
        log_csv(log_str.split(";"), file_name=self.ds_name)
        model = self.model_conv()

        optimizer = tf.keras.optimizers.Adam()
        loss_value = 0
        index = 0
        kmeans_n_init = 100
        assignment = np.array([-1] * len(x))
        index_array = np.arange(x.shape[0])
        for ite in range(int(140 * 100)):
            if ite % params["update_interval"] == 0:
                H = model(x).numpy()[:, : self.hidden_units]
                ans_kmeans = KMeans(
                    n_clusters=self.n_clusters, n_init=kmeans_n_init
                ).fit(H)
                kmeans_n_init = int(ans_kmeans.n_iter_ * 2)
                # log the clustering params on neptune
                run["kmeans_summary"] = npt_utils.create_kmeans_summary(
                    ans_kmeans, H, n_clusters=self.n_clusters
                )

                U = ans_kmeans.cluster_centers_
                assignment_new = ans_kmeans.labels_

                w = np.zeros((self.n_clusters, self.n_clusters), dtype=np.int64)
                for i, j in enumerate(assignment_new):
                    w[assignment_new[i], assignment[i]] += 1
                ind = linear_assignment(-w)
                temp = np.array(assignment)
                for i in range(self.n_clusters):
                    assignment[temp == ind[1][i]] = i
                n_change_assignment = np.sum(assignment_new != assignment)
                assignment = assignment_new

                S_i = []
                for i in range(self.n_clusters):
                    temp = H[assignment == i] - U[i]
                    temp = np.matmul(np.transpose(temp), temp)
                    S_i.append(temp)
                S_i = np.array(S_i)
                S = np.sum(S_i, 0)
                _Evals, V = self.sorted_eig(S)
                H_vt = np.matmul(H, V)  # 1000,5
                U_vt = np.matmul(U, V)  # 10,5
                #
                loss = np.round(np.mean(loss_value), 5)
                acc, nmi, fms, hcv, mis, rs = get_metrics(
                    np.array(y), np.array(assignment)
                )

                # log to neptune
                run["metric/acc"].log(acc)
                run["metric/nmi"].log(nmi)
                run["metric/fms"].log(fms)
                run["metric/hcv"].log(hcv)
                run["metric/mis"].log(mis)
                run["metric/rs"].log(rs)

                # log
                log_str = (
                    f'iter {ite // params["update_interval"]}; acc, nmi,fms,hcv,mis,rs, ri = {acc, nmi,fms,hcv,mis,rs, loss}; loss:'
                    f"{loss:.5f}; n_changed_assignment:{n_change_assignment}; time:{time.time() - time_start:.3f}"
                )
                print(log_str)
                log_csv(log_str.split(";"), file_name=self.ds_name)

            if n_change_assignment <= len(x) * 0.005:
                model.save_weights(f"log_history/weight_final_l2_{self.ds_name}.h5")
                print("end")
                break
            idx = index_array[
                index
                * params["batch_size"]: min(
                    (index + 1) * params["batch_size"], x.shape[0]
                )
            ]
            y_true = H_vt[idx]
            temp = assignment[idx]
            for i in range(len(idx)):
                y_true[i, -1] = U_vt[temp[i], -1]

            with tf.GradientTape() as tape:
                tape.watch(model.trainable_variables)
                y_pred = model(x[idx])
                y_pred_cluster = tf.matmul(y_pred[:, : self.hidden_units], V)
                loss_value = losses.mse(y_true, y_pred_cluster)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            index = index + 1 if (index + 1) * params["batch_size"] <= x.shape[0] else 0
            # print('This is model 2')
