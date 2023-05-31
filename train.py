import flax.linen as nn
import flax
import tensorflow as tf
import numpy as np
# from sklearn.preprocessing import OneHotEncoder

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

class train():
    def __init__(self,x_train, y_train, model, learning_rate, batch_size=10):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.optimizer = flax.optim.Adam(learning_rate = learning_rate).create(model)
        self.batch_size = batch_size

    def one_hot_encoder(self,y_in):


    def _cross_entropy_loss(self,y_pred, y_train):
        # retrieves the number of dims of y predicted
        num_examples = y_pred.shape[0]
        y_true_onehot = self.one_hot_encoder(y_pred)



    def loss_fn(self,batch):
        model = self.model
        y_pred = model(batch["x"])
        loss = self._cross_entropy_loss(y_pred,y_train)

    def train_step(self,batch):
