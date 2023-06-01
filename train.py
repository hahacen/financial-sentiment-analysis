import flax.linen as nn
import flax
import jax
import tensorflow as tf
import numpy as np
import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
import tensorflow_probability.python.internal.backend.jax.nn as jax_nn
from flax.linen import compact
import jax.numpy as jnp
import meta_parameters
import helper
from googletrans import Translator
from deep_translator import GoogleTranslator
from collections import defaultdict

# define the network
class MLP(nn.Module):

    def setup(self):
        # super().__init__()
        # kernel_init = nn.initializers.xavier_uniform()
        self.dense1 = nn.Dense(features=4)
        self.dense2 = nn.Dense(features=3)
        # self.params = self.param("*", init_fn=nn.initializers.xavier_uniform())

        # self._input_dim = input_dim
        # self._output_dims = output_dims
    @nn.compact
    def __call__(self, x):
        x = self.dense1(x)
        x = nn.relu(x)  # Apply ReLU activation using jax.nn.relu
        return self.dense2(x)


class trainer:
    # score shall be a list of tuple
    @compact
    def __init__(self, train_csv_path,
                 model, learning_rate=meta_parameters.learning_rate, batch_size=10):
        self.model = model
        # manually parse x and y for train
        self.preprocess(train_csv_path)
        # var = model.init(meta_parameters.rng, self._x_train)
        # Initialize the parameters of the model
        input_shape = (self._x_train.shape[0], 1)  # Example input shape
        initial_params = model.init(meta_parameters.rng, jnp.ones(input_shape))
        self.optimizer = flax.optim.Adam(learning_rate=learning_rate).create(initial_params)
        self.batch_size = batch_size

    def preprocess(self, csv_path):
        train_csv = helper.parsing(csv_path)
        df = pd.read_csv(train_csv)
        print(df)
        x_train_descriptions = df['description'].to_numpy()
        x = []
        # translator = Translator()
        translator = GoogleTranslator(source='auto', target='en')
        count = 0
        for description in x_train_descriptions:
            if len(description) > 100:
                description = description[:100]
            text0 = translator.translate(description)
            print(text0)  # for debug use
            print(description)
            # print(count)
            x_num = helper.score_tuple(text0)
            print(x_num)
            # print(description)
            count = count +1
            if count > 5:
                break
            x.append(x_num)

        y = self.one_hot_encoder(df['label'].to_numpy())
        print(y)
        self._x_train = np.array(x)
        self._y_train = y
        return self._x_train, self._y_train

    # this encoder deal with all the sample data
    # shape: sample_size, str
    def one_hot_encoder(self, y_in) -> np.array:
        helper_dic = {
            "强裂推荐": 1,
            "强推": 1,
            "谨慎推荐": 0,
            "中性": 0,
            "sell": -1,
            "卖出": -1,
            "SELL": -1,
            "Neutral": 0,
            "减持": -1,
            "Reduce": -1
        }

        # this is the one_hot two-dimensional array of one batch
        # 3 is the num of categories
        one_hot = np.zeros((y_in.shape[0], 3))
        # print(one_hot)
        for idx in range(y_in.shape[0]):
            # set the index of dictionary to 1
            one_hot[idx][helper_dic[y_in[idx]] + 1] = 1
        return one_hot
    # Each step will take in a batch of information
    # @jax.jit
    def train_step(self, batch):
        def loss_fn(model):
            y_pred_logits = model(batch["x"])
            loss = jax.nn.softmax_cross_entropy_with_logits(batch["y"], y_pred_logits)
            avg_loss = jnp.mean(loss)
            return avg_loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(self.model)
        self.optimizer = self.optimizer.apply_gradient(grad)
        return loss

    def _data_iterator(self):
        num_samples = self._x_train.shape[0]
        print(self._y_train.shape)
        num_batches = num_samples
        while True:
            indices = np.random.permutation(num_samples)
            x = self._x_train[indices]
            y = self._y_train[indices]

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size
                yield {"x": x[start_idx:end_idx], "y": y[start_idx:end_idx]}

    def train(self, num_epochs):
        data_iter = self._data_iterator()
        total_steps = self._x_train.shape[0]
        # print(self._x_train)
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for step in range(1, total_steps+1):
                batch = next(data_iter)
                # print(batch['y'])
                loss = self.train_step(batch)
                epoch_loss += loss
            print(f"Epoch {epoch + 1} Loss: {epoch_loss / total_steps}")
