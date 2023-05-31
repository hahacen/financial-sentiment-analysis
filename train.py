import flax.linen as nn
import flax
import jax
import tensorflow as tf
import numpy as np
import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
import tensorflow_probability.python.internal.backend.jax.nn as jax_nn
import tensorflow as tf
# define the network
class MLP(nn.Module):
    def __init__(self, input_dim, output_dims):
        self.dense1 = nn.Dense(features = 8, kernal_init = nn.initializers.xavier_uniform(),activation = nn.relu)
        self.dense2 = nn.Dense(features = 5, kernel_init = nn.initializers.xavier_uniform(),activation = nn.softmax)
        self._input_dim = input_dim
        self._output_dims = output_dims

    def __call__(self, x):
        x = self.dense1(x)
        return self.dense2(x)

class trainer():
    def __init__(self,train_csv_path, model, learning_rate=0.01, batch_size=10):
        self.model = model
        # TODO: manually parse x and y for train
        self.preprocess(train_csv_path)
        self.optimizer = flax.optim.Adam(learning_rate = learning_rate).create(model)
        self.batch_size = batch_size


    def preprocess(self,csv_path):
        df = pd.read_csv(csv_path)
        temp_xTrain = df['TITLE']
        temp_yTrain = df[df['ORGAN_RATING_CONTENT'].isin(['强裂推荐','强推','谨慎推荐','中性',
                                                          'sell','卖出','SELL','Neutral','减持','Reduce'])]
        y_t=temp_yTrain['ORGAN_RATING_CONTENT']
        x_train = temp_xTrain.tolist()
        y_train = self.one_hot_encoder(y_t.tolist())
        self._x_train = x_train
        self._y_train = y_train
        return x_train,y_train

    def one_hot_encoder(self,y_in)->np.list:
        helper_dic = {
            "强裂推荐":1,
            "强推":1,
            "谨慎推荐":0,
            "中性":0,
            "sell":-1,
            "卖出":-1,
            "SELL":-1,
            "Neutral":0,
            "减持":-1,
            "Reduce":-1
        }
        one_hot = np.zeros(3)
        one_hot[helper_dic[]] = 1

    # Define the cross-entropy loss function
    def cross_entropy_loss(self,y_pred, y_true):
        num_examples = y_pred.shape[0]
        y_true_onehot = self.one_hot_encoder(y_pred)
        loss = -np.mean(y_true_onehot * np.log(y_pred + 1e-10))
        return loss

    def train_step(self,batch):
        def loss_fn(model):
            y_pred = model(batch["x"])
            #TODO:
            loss = tf.nn.softmax_cross_entropy_with_logits()
            return loss
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(self.optimizer.target)
        self.optimizer = self.optimizer.apply_gradient(grad)
        return loss

    def _data_iterator(self):
        num_samples = self._x_train.shape[0]
        num_batches = num_samples
        while True:
            indices = np.random.permutation(num_samples)
            x = self.x_train[indices]
            y = self.y_train[indices]

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size
                yield {"x": x[start_idx:end_idx], "y": y[start_idx:end_idx]}
    def train(self, num_epochs):
        data_iter = self._data_iterator()
        total_steps = self._x_train.shape[0]

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for step in range(total_steps):
                batch = next(data_iter)
                loss = self.train_step(batch)
                epoch_loss += loss
            print(f"Epoch {epoch + 1} Loss: {epoch_loss / total_steps}")
