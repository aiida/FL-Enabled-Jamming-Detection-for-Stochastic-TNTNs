# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:59:10 2024

@author: AP93300
"""


import collections
import itertools
import os
import time

import attr
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nest_asyncio

import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf  # 2.9.3
import tensorflow_federated as tff
import tensorflow_probability as tfp
from numpy import inf, random

# nest_asyncio.apply()

tfd = tfp.distributions

import seaborn as sns
from PIL import Image
from scipy import stats
from sklearn import metrics
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from autoencoder_model import encoder_fn, decoder_fn

#print(tf.__version__)
#print(tff.__version__)
#print(np.__version__)
#print(tfp.__version__)
#print(np.__version__)

# Settings

# number of classes = number of types of waveforms
NUM_CLASSES = 2

# Federated learning settings
NUM_CLIENTS = 10
NUM_CLIENT_PER_ROUND = 4

NUM_ROUNDS = 10

CLIENT_LR = 0.05
SERVER_LR = 1.0

# Local model settings
NUM_EPOCHS = 30  # 20, 50
BATCH_SIZE = 32

NON_FL_LOCAL_EPOCH = 10
NON_FL_LOCAL_BACTH_SIZE = 32

# Other settings
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

latent_dim = 180

## Train data: Feature SCF Set

def Load_wifi_file(path_file):
  data_load = sio.loadmat(path_file)
  wifi_data = data_load["wifi_data"]

  # Pre-processing the data
  wifi_data = np.log10(wifi_data)
  wifi_data[wifi_data == -inf] = 0

  return wifi_data

def Load_lte_file(path_file):
  data_load = sio.loadmat(path_file)
  lte_data = data_load["lte_data"]

  # Pre-processing the data
  lte_data = np.log10(lte_data)
  lte_data[lte_data == -inf] = 0

  return lte_data

### Load data

# Define dictionaries to store data
wifi_data = {}
lte_data = {}
input_data = {}

# Load WiFi and LTE data for each CH using loops
for i in range(1, 11):
    wifi_data[f'CH{i}'] = Load_wifi_file(f"./unjammed_jammed_mobile_wifi_data_CH{i}.mat")
    lte_data[f'CH{i}'] = Load_lte_file(f"./unjammed_jammed_mobile_lte_data_CH{i}.mat")

# Concatenate and normalize data for each CH using loops
for i in range(1, 11):
    wifi_ch = wifi_data[f'CH{i}']
    lte_ch = lte_data[f'CH{i}']
    concatenated_data = np.concatenate((wifi_ch, lte_ch), axis=0)
    min_val = tf.reduce_min(concatenated_data)
    max_val = tf.reduce_max(concatenated_data)
    normalized_data = (concatenated_data - min_val) / (max_val - min_val)
    input_data[f'CH{i}'] = tf.cast(normalized_data, tf.float32)


## Label Set

# Define dictionaries to store label data
wifi_labels = {}
lte_labels = {}
x_input_data = {}
y_input_data = {}

# Load WiFi and LTE labels for each CH using loops
for i in range(1, 11):
    wifi_label_load = sio.loadmat(f"./unjammed_jammed_mobile_wifi_label_CH{i}.mat")
    wifi_labels[f'CH{i}'] = wifi_label_load["wifi_label"]
    
    lte_label_load = sio.loadmat(f"./unjammed_jammed_mobile_lte_label_CH{i}.mat")
    lte_labels[f'CH{i}'] = lte_label_load["lte_label"]

# Concatenate and reshape labels for each CH using loops
for i in range(1, 11):
    wifi_label_ch = wifi_labels[f'CH{i}']
    lte_label_ch = lte_labels[f'CH{i}']
    concatenated_labels = np.concatenate((wifi_label_ch, lte_label_ch), axis=0)
    reshaped_labels = np.ravel(concatenated_labels)
    y_input_data[f'CH{i}'] = reshaped_labels

# Shuffle data for each CH using loops
for i in range(1, 11):
    index = np.arange(input_data[f'CH{i}'].shape[0])
    np.random.shuffle(index)
    x_input_data[f'CH{i}'] = input_data[f'CH{i}'][index, :]
    y_input_data[f'CH{i}'] = y_input_data[f'CH{i}'][index]


print(x_input_data['CH1'].shape, y_input_data['CH1'].shape)


# Define dictionaries to store train and test data
train_data = {}
test_data = {}

# Define test sizes for each CH
test_sizes = [0.19, 0.10, 0.12, 0.13, 0.17, 0.12, 0.11, 0.14, 0.13, 0.25]

# Perform train-test split for each CH using loops
for i in range(1, 11):
    x_input_CH = locals()[f'x_input_CH{i}']
    y_input_CH = locals()[f'y_input_CH{i}']
    x_train_CH, x_test_CH, y_train_CH, y_test_CH = train_test_split(x_input_CH, y_input_CH, test_size=test_sizes[i-1], random_state=13)
    train_data[f'CH{i}'] = (x_train_CH, y_train_CH)
    test_data[f'CH{i}'] = (x_test_CH, y_test_CH)

# Access and print train and test data for each CH using loops
for i in range(1, 11):
    print(f"CH{i} Train Data:")
    print(train_data[f'CH{i}'][0])  # Print x_train_CH{i}
    print(train_data[f'CH{i}'][1])  # Print y_train_CH{i}
    print(f"CH{i} Test Data:")
    print(test_data[f'CH{i}'][0])   # Print x_test_CH{i}
    print(test_data[f'CH{i}'][1])   # Print y_test_CH{i}


# Define dictionaries to store normal and anomalous train data for each CH
normal_train_data = {}
anomalous_train_data = {}

# Loop through each CH
for i in range(1, 11):
    y_train_CH = locals()[f'y_train_CH{i}'].astype(bool)
    x_train_CH = locals()[f'x_train_CH{i}']
    
    # Split into normal and anomalous train data
    normal_train_data[f'CH{i}'] = x_train_CH[~y_train_CH], y_train_CH[~y_train_CH].astype(int)
    anomalous_train_data[f'CH{i}'] = x_train_CH[y_train_CH], y_train_CH[y_train_CH].astype(int)

# Print the lengths of normal train data for each CH
for ch, data in normal_train_data.items():
    print(f"CH {ch} Normal Train Data Length: {len(data[0])}")


# Define dictionaries to store normal and anomalous test data for each CH
normal_test_data = {}
anomalous_test_data = {}

# Loop through each CH
for i in range(1, 11):
    y_test_CH = locals()[f'y_test_CH{i}'].astype(bool)
    x_test_CH = locals()[f'x_test_CH{i}']
    
    # Split into normal and anomalous test data
    normal_test_data[f'CH{i}'] = x_test_CH[~y_test_CH], y_test_CH[~y_test_CH].astype(int)
    anomalous_test_data[f'CH{i}'] = x_test_CH[y_test_CH], y_test_CH[y_test_CH].astype(int)

# Print the lengths of normal test data for each CH
for ch, data in normal_test_data.items():
    print(f"CH {ch} Normal Test Data Length: {len(data[0])}")


print("No. of records of jammed Train Data in CH1=", len(anomalous_train_data['CH1'][0]))
print("No. of records of unjammed Train data in CH1=", len(normal_train_data['CH1'][0]))

print("No. of records of jammed Test Data in CH1=", len(anomalous_test_data['CH1'][0]))
print("No. of records of unjammed Test data in CH1=", len(normal_test_data['CH1'][0]))


####### Federated Train dataset
NUM_CLIENTS = 10  # Define the number of clients

client_train_dataset = collections.OrderedDict()
client_train_dataset_predict = collections.OrderedDict()
client_abnormal_train_dataset = collections.OrderedDict()

for client_id in range(1, NUM_CLIENTS + 1):
    client_train_dataset[f'client_{client_id}'] = collections.OrderedDict()
    client_train_dataset_predict[f'client_{client_id}'] = collections.OrderedDict()
    client_abnormal_train_dataset[f'client_{client_id}'] = collections.OrderedDict()

    # Normal train dataset
    total_image_count = len(globals()[f'normal_train_CH{client_id}'])
    idx = np.arange(total_image_count)

    feature1 = []
    feature2 = []
    labels = []

    for k in idx:
        feature1.append(globals()[f'normal_train_CH{client_id}'][k])
        feature2.append(globals()[f'normal_train_CH{client_id}'][k])
        labels.append(globals()[f'normal_train_label_CH{client_id}'][k])

    data = collections.OrderedDict((('pixels1', np.array(feature1)),
                                    ('pixels2', np.array(feature2))))
    data_predict = collections.OrderedDict((('label', np.array(labels)),
                                            ('pixels', np.array(feature1))))

    client_train_dataset[f'client_{client_id}'] = data
    client_train_dataset_predict[f'client_{client_id}'] = data_predict

    # Abnormal train dataset
    total_abnormal_image_count = len(globals()[f'anomalous_train_CH{client_id}'])
    abn_idx = np.arange(total_abnormal_image_count)

    feature_abnormal = []
    label_abnormal = []

    for k in abn_idx:
        feature_abnormal.append(globals()[f'anomalous_train_CH{client_id}'][k])
        label_abnormal.append(globals()[f'anomalous_train_label_CH{client_id}'][k])

    abnormal_data = collections.OrderedDict((('label', np.array(label_abnormal)),
                                             ('pixels', np.array(feature_abnormal))))

    client_abnormal_train_dataset[f'client_{client_id}'] = abnormal_data

# Convert to TestClientData
signal_train = tff.simulation.datasets.TestClientData(client_train_dataset)
signal_train_predict = tff.simulation.datasets.TestClientData(client_train_dataset_predict)
signal_abnormal_train = tff.simulation.datasets.TestClientData(client_abnormal_train_dataset)

# Create TF datasets
train_dataset = signal_train.create_tf_dataset_for_client(signal_train.client_ids[0])
train_dataset_predict = signal_train_predict.create_tf_dataset_for_client(signal_train_predict.client_ids[0])
train_abnormal_dataset = signal_abnormal_train.create_tf_dataset_for_client(signal_abnormal_train.client_ids[0])

# Example elements
example_element = next(iter(train_dataset))
example_element_predict = next(iter(train_dataset_predict))
example_abnormal_element = next(iter(train_abnormal_dataset))

print('done')

# Number of examples per layer for a sample of clients
classes = ['unjammed', 'jammed']
for client_id in range(1, 4):  # Sample clients
    client_dataset = signal_train_predict.create_tf_dataset_for_client(f'client_{client_id}')
    plot_data = collections.defaultdict(list)
    for example in client_dataset:
        label = example['label'].numpy()
        plot_data[label].append(label)

    for j in range(2):
        print(f'CH {client_id}', 'waveform', classes[j], len(plot_data[j]))


#### Federated Test dataset

##################### test dataset ######################

client_test_dataset = collections.OrderedDict()
client_test_dataset_predict = collections.OrderedDict()

for client_id in range(1, 11):  # Assuming 10 clients
    data, data_predict, test_data_CH = fed_test_data(
        globals()[f'normal_test_label_CH{client_id}'],
        globals()[f'anomalous_train_label_CH{client_id}'],
        globals()[f'anomalous_test_label_CH{client_id}'],
        globals()[f'normal_test_CH{client_id}'],
        globals()[f'anomalous_train_CH{client_id}'],
        globals()[f'anomalous_test_CH{client_id}']
    )

    client_test_dataset[f'client_{client_id}'] = data
    client_test_dataset_predict[f'client_{client_id}'] = data_predict

signal_test = tff.simulation.datasets.TestClientData(client_test_dataset)
signal_test_predict = tff.simulation.datasets.TestClientData(client_test_dataset_predict)

test_dataset = signal_test.create_tf_dataset_for_client(signal_test.client_ids[0])
test_dataset_predict = signal_test_predict.create_tf_dataset_for_client(signal_test_predict.client_ids[0])

example_element = next(iter(test_dataset))
example_element_predict = next(iter(test_dataset_predict))

print('done')

## Preprocess the federated datasets (train dataset)

def preprocess(dataset):
    def batch_format_fn(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['pixels1'], [-1, 33, 65, 1])),
            ('y', tf.reshape(element['pixels2'], [-1, 33, 65, 1])),

        ])

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

def preprocess_abnormal(dataset):
    def batch_format_fn(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['pixels'], [-1, 33, 65, 1])),
            ('y', tf.reshape(element['label'], [-1, 33, 65, 1])),

        ])

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

def preprocess_test_dataset(dataset):
    def batch_format_fn(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['pixels1'], [-1, 33, 65, 1])),
            ('y', tf.reshape(element['pixels2'], [-1, 33, 65, 1])),
        ])

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
        128).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

def make_federated_test_data(client_data, client_ids):
    return [preprocess_test_dataset(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

def make_federated_abnormal_data(client_data, client_ids):
    return [preprocess_abnormal(client_data.create_tf_dataset_for_client(x)) for x in client_ids]


preprocessed_train_dataset = preprocess(train_dataset)


sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                     next(iter(preprocessed_train_dataset)))

federated_train_data = make_federated_data(signal_train,
                                           signal_train.client_ids)

federated_abnormal_data = make_federated_abnormal_data(signal_abnormal_train,
                                           signal_abnormal_train.client_ids)

print('done preparing federated data')

# Training Phase

@attr.s(eq=False, frozen=True, slots=True)
class ClientOutput:
    weights_delta = attr.ib()
    client_weight = attr.ib()
    model_output = attr.ib()

@attr.s(eq=False, frozen=True, slots=True)
class ServerState(object):
    trainable_weights = attr.ib()
    optimizer_state = attr.ib()

# At Local Client Side
## Build local AutoEncoder Network


# Define the input layer for the VAE
def vae_fn():
    # Create the encoder and decoder
    encoder = encoder_fn()
    decoder = decoder_fn()
    input_layer = tf.keras.layers.Input(shape=(33, 65, 1))

    # Connect the input to the encoder
    z_mean, z_log_var, z = encoder(input_layer)

    # Connect the decoder to the latent space
    vae_output = decoder(z)

    # Create the VAE model
    vae = tf.keras.models.Model(input_layer, vae_output, name='vae')
    return vae

model = vae_fn()
print(model.summary())

# At Global PS side
## AE model at PS

import tensorflow as tf

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, mse_weight=0.000, kl_weight=0.001, **kwargs):   #0.001
        super().__init__(**kwargs)
        self.mse_weight = mse_weight
        self.kl_weight = kl_weight

    def call(self, y_true, y_pred):
        # Calculate the Mean Squared Error (MSE) loss
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)

        num_latent_variables = 33 * 65  # Calculate the total number of latent variables
        y_pred = tf.reshape(y_pred, [-1, num_latent_variables * 2])

        # Split y_pred into mean and log variance components
        mean, log_var = tf.split(y_pred, num_or_size_splits=2, axis=-1)

        # Calculate the Kullback-Leibler Divergence (KL Divergence) loss
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1)
        kl_loss = tf.reduce_mean(kl_loss)


        # Combine the MSE and KL Divergence losses with weights
        combined_loss = self.mse_weight * mse_loss + self.kl_weight * kl_loss

        return combined_loss


# attributes : trainable_variables & non_trainable_variables
# @tff.tf_computation
def model_fn():
    autoencoder_model = vae_fn() #VAE_fn() #autoencoder_fn()
    federated_model = tff.learning.models.from_keras_model(
        autoencoder_model,
        input_spec=federated_train_data[0].element_spec,
        loss = CustomLoss(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    return federated_model

aida_model = model_fn()

unfinalized_metrics_type = tff.types.type_from_tensors(aida_model.report_local_unfinalized_metrics())
metrics_aggregation_computation = tff.learning.metrics.sum_then_finalize(aida_model.metric_finalizers(), unfinalized_metrics_type)

## Functions for Aggregation at Parameter Server Side

@attr.s(eq=False, frozen=True, slots=True)
class ServerState(object):
    trainable_weights = attr.ib()
    optimizer_state = attr.ib()

@tf.function
def server_update(server_state, mean_model_delta, server_optimizer):
    """Updates the server model weights."""
    # Use aggregated negative model delta as pseudo gradient.
    negative_weights_delta = tf.nest.map_structure(lambda w: -1.0 * w, mean_model_delta) #-1.0
    new_optimizer_state, updated_weights = server_optimizer.next(
        server_state.optimizer_state,
        server_state.trainable_weights,
        negative_weights_delta,
    )
    return tff.structure.update_struct(
        server_state,
        trainable_weights=updated_weights,
        optimizer_state=new_optimizer_state,
    )


### 0.01 / 0.1 / 1.0
server_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=1.0)


# 2. Functions return initial state on server.
@tff.tf_computation
def server_init():
    # Create the model and return its trainable variables
    model = model_fn()
    trainable_tensor_specs = tf.nest.map_structure(lambda v: tf.TensorSpec(v.shape, v.dtype), model.trainable_variables)
    optimizer_state = server_optimizer.initialize(trainable_tensor_specs)
    return ServerState(trainable_weights=model.trainable_variables, optimizer_state=optimizer_state)


@tff.federated_computation
def server_init_tff():
    return tff.federated_value(server_init(), tff.SERVER)


aida_model = model_fn()
tf_dataset_type = tff.SequenceType(aida_model.input_spec)

server_state_type = server_init.type_signature.result

trainable_weights_type = server_state_type.trainable_weights

# server update
@tff.tf_computation(server_state_type, trainable_weights_type)
def server_update_fn(server_state, model_delta):
    # model = model_fn()
    return server_update(server_state, model_delta, server_optimizer)


## Functions for Local Update at Client Side during Aggregation

@tf.function
def client_update(model, dataset, server_weights, client_optimizer):
    """Performs training (using the server model weights) on the client's dataset."""
    # Initialize the client model with the current server weights.
    client_weights = model.trainable_variables
    # Assign the server weights to the client model.
    tf.nest.map_structure(lambda v, t: v.assign(t),
                          client_weights, server_weights)

    # Use the client_optimizer to update the local model.
    num_examples = 0.0

    for batch in dataset:
        with tf.GradientTape() as tape:
            # Compute a forward pass on the batch of data
            outputs = model.forward_pass(batch)

            num_examples += tf.cast(outputs.num_examples, tf.float32)

            # Compute the corresponding gradient
            grads = tape.gradient(outputs.loss, client_weights)

            grads_and_vars = zip(grads, client_weights)

        # Apply the gradient using a client optimizer.
        client_optimizer.apply_gradients(grads_and_vars)

    # Compute the difference between the server weights and the client weights
    client_update = tf.nest.map_structure(tf.subtract,
                                          client_weights,
                                          server_weights)

    client_weight = num_examples

    # Aggregate the KDE weights over all batches

    model_outputs = model.report_local_unfinalized_metrics()

    return ClientOutput(client_update, client_weight, model_outputs)


# 3. One round of computation and communication.
server_state_type = server_init.type_signature.result
# print('server_state_type:\n',
#       server_state_type.formatted_representation())
trainable_weights_type = server_state_type.trainable_weights
# print('trainable_weights_type:\n',
#       trainable_weights_type.formatted_representation())


@tff.tf_computation(tf_dataset_type, trainable_weights_type)
def client_update_fn(tf_dataset, server_weights):
    model = model_fn()
    # Add learning rate scheduling
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.004,
        decay_steps=1000,
        decay_rate=0.9)
    client_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule) #0.004
    return client_update(model, tf_dataset, server_weights, client_optimizer)

# 3-2. Orchestration with `tff.federated_computation`.
federated_server_type = tff.FederatedType(server_state_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

# In[]
@tff.federated_computation
def initialize_fn():
    return tff.federated_value(server_init(), tff.SERVER)

@tff.federated_computation(federated_server_type, federated_dataset_type)
# function next: one round of communication
def next_fn(server_state, federated_dataset):
    
    # Broadcast the server weights to the clients.
    server_weights_at_client = tff.federated_broadcast(server_state.trainable_weights)

    # Each client computes their updated weights.
    # client_weights = client_update(federated_dataset, server_weights_at_client)
    client_weights = tff.federated_map(client_update_fn, (federated_dataset, server_weights_at_client))

    weight_denom = client_weights.client_weight

    # The server averages these updates.
    mean_client_weights = tff.federated_mean(client_weights.weights_delta, weight=weight_denom)

    # The server updates its model.
    # server_weights = server_update(mean_client_weights)
    server_weights = tff.federated_map(server_update_fn, (server_state, mean_client_weights))

    aggregated_outputs = metrics_aggregation_computation(client_weights.model_output)

    return server_weights, aggregated_outputs

# In[]
iterative_process = tff.templates.IterativeProcess(
    initialize_fn=initialize_fn,
    next_fn=next_fn
    )

## Aggregation loop (both)

import time

"""Trains the federated averaging process and output metrics."""

acc_fl = []  # during training (like in cross-validation of keras)
loss_fl = []  # during training
time_fl = []  # during training

mode = "random"  # client selection model

NUM_ROUNDS = 10
NUM_CLIENT_PER_ROUND = 8

step = 0
Nb = 1  # the number of PS in the area

# Initialize the Federated Averaging algorithm to get the initial server state.
state = iterative_process.initialize()

# Train the model using Federated Learning
for round_num in range(NUM_ROUNDS):
    t1 = time.time()

    # Sample the clients parcitipated in this round.
    if mode == "random":
        sampled_clients = np.random.choice(signal_train.client_ids, size=NUM_CLIENT_PER_ROUND, replace=False)

    elif mode == "rrobin":
        # step!!!!
        nodes = signal_train.client_ids
        node_index = (step * NUM_CLIENT_PER_ROUND) % NUM_CLIENTS

        if node_index + NUM_CLIENT_PER_ROUND <= NUM_CLIENTS:
            sampled_clients = nodes[node_index : node_index + NUM_CLIENT_PER_ROUND]
        else:
            sampled_clients = nodes[node_index:] + nodes[: node_index + NUM_CLIENT_PER_ROUND - NUM_CLIENTS]

    elif mode == "prop_k":
        sampled_clients = []
        h = tf.convert_to_tensor(random.exponential(scale=1, size=[Nb, NUM_CLIENTS]))
        nodes = np.argsort(h.numpy()[0, :])[-NUM_CLIENT_PER_ROUND:]
        for i in range(len(nodes)):
            sampled_clients.append(signal_train.client_ids[nodes[i]])

    else:
        sampled_clients = signal_train.client_ids

    # Create a list of `tf.Dataset` instances from the data of sampled clients.
    sampled_train_data = [preprocess(signal_train.create_tf_dataset_for_client(client)) for client in sampled_clients]

    # Round one round of the algorithm based on the server state and client data
    # and output the new state and metrics.

    # Perform one round of Federated Learning
    state, metrics = iterative_process.next(state, sampled_train_data)

    # print (state)
    t2 = time.time()
    # Print the results for this round
    print("round {:2d}, metrics={}, " "round time: {t:.2f} seconds".format(round_num, metrics, t=t2 - t1))

    acc_fl.append(metrics["categorical_accuracy"])

    loss_fl.append(metrics["loss"])

    time_fl.append(t2 - t1)

print(loss_fl)
print(time_fl)


# Testing (Inference) Phase

## Step 1:  Preprocess the federated Test data

def preprocess_prediction(dataset):
    def batch_format_fn(element):
        return collections.OrderedDict(
            [
                (
                    "x",
                    tf.reshape(
                        element["pixels"],
                        [-1, 33, 65, 1],
                    ),
                ),
                ("y", tf.reshape(element["label"], [-1, 1])),
            ]
        )

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(128).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


def make_federated_prediction(client_data, client_ids):
    return [preprocess_prediction(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

federated_test_prediction = make_federated_prediction(signal_test_predict, signal_test_predict.client_ids)

state

## Step 2: Calculate the reconstruction error of the train dataset`federated_train_data`

print("[INFO] ----------- making predictions of the AutoEncoder---------------")
print("Detect anomalies with reconstruction error")

# to not show the training information like ms/step
tf.keras.utils.disable_interactive_logging()

pred = []  # this vector will contains the predicted features of the encoder phase
true_data = []  # this vector will contains the original features of the federated_train_data

trained_global_model = None
trained_global_model = vae_fn()
#print(state)
# trained_global_model

tf.nest.map_structure(lambda v, t: v.assign(t),
                      trained_global_model.trainable_variables,
                      list(state.trainable_weights))


# calculate the reconstruction error for normal train

for batch in federated_train_data:
    for feature in batch:
        x_pred = trained_global_model.predict(feature["x"])
        xx_pred = np.array(x_pred).reshape(x_pred.shape[0], x_pred.shape[1], x_pred.shape[2])
        pred.append(xx_pred)
        true_data.append(np.array(feature["x"]).reshape(feature["x"].shape[0], feature["x"].shape[1], feature["x"].shape[2]))


normal_true = []
normal_true = [i for j in true_data for i in j]

normal_pred = []
normal_pred = [i for j in pred for i in j]

normal_true = np.array(normal_true)
normal_true = normal_true.reshape(normal_true.shape[0], normal_true.shape[1], normal_true.shape[2])

normal_pred = np.array(normal_pred)
normal_pred = normal_pred.reshape(normal_pred.shape[0], normal_pred.shape[1], normal_pred.shape[2])


cross_ent_from_train_data = tf.keras.backend.binary_crossentropy(normal_true, normal_pred)
train_normal_reconstruction_error = tf.reduce_sum(cross_ent_from_train_data, axis=[1,2]) #consolidate at each instance
print(train_normal_reconstruction_error.shape)

abn_pred = []
abn_true = []
abn_label =[]

# to not show the training information like ms/step
tf.keras.utils.disable_interactive_logging()

temp_num_batch = 0
temp_num_samples = 0
for batch in federated_abnormal_data:
    temp_num_batch += 1
    for feature in batch:
        temp_num_samples += 1
        x_pred = trained_global_model.predict(feature['x'])
        x_pred = np.array(x_pred).reshape(x_pred.shape[0], x_pred.shape[1], x_pred.shape[2])
        abn_pred.append(x_pred)
        abn_true.append(np.array(feature['x']).reshape(feature['x'].shape[0], feature['x'].shape[1], feature['x'].shape[2]))
        abn_label.append(feature['y'])


print(x_pred.shape)
print(temp_num_batch)
print(temp_num_samples)
print(len(abn_pred))
print(abn_pred[-1].shape)

true_abn = []
true_abn = [i for j in abn_true for i in j]

pred_abn = []
pred_abn = [i for j in abn_pred for i in j]

y_true = []
y_true = [i for j in abn_label for i in j]

true_abn = np.array(true_abn)
true_abn = true_abn.reshape(true_abn.shape[0], true_abn.shape[1], true_abn.shape[2])

pred_abn = np.array(pred_abn)
pred_abn = pred_abn.reshape(pred_abn.shape[0], pred_abn.shape[1], pred_abn.shape[2])

y_true=np.array(y_true)
y_true=np.argmax(y_true, axis=-1)  # do not need, because using sparse entropy, just integer

print(y_true.shape)

cross_ent_from_abnormal_data = tf.keras.backend.binary_crossentropy(true_abn, pred_abn)

print(cross_ent_from_abnormal_data.shape)

# calculate the reconstruction error for test dataset
train_abnormal_reconstruction_error = tf.reduce_sum(cross_ent_from_abnormal_data, axis=[1,2]) #consolidate at each instance

print(train_abnormal_reconstruction_error.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
#plot the reconstruction error for train normal and anomaly data
plt.plot(train_normal_reconstruction_error[:400], "o", color='teal', label = 'federated_train_data')
plt.plot(train_abnormal_reconstruction_error[:400], "o", color='brown', label = 'federated_abnormal_data')
# plt.plot(test_reconstruction_error[:100], 'g-', label = 'federated_test_prediction')

#plt.ylabel('Reconstruction Error')
plt.legend()


## Step 4: Compute decision threshold

#The decision threshold is computed based on the `train_normal_reconstruction_error` only

decision_threshold = round(np.percentile(train_normal_reconstruction_error, 99),3)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))

# Use different variables for sns.kdeplot objects
plot1 = sns.kdeplot(train_normal_reconstruction_error, fill=True, color='teal',
                  label='Unjammed (train, normal) reconstruction error')

plot2 = sns.kdeplot(train_abnormal_reconstruction_error, fill=True, color='brown',
                  label='Jammed (train, abnormal) reconstruction error')

plt.axvline(decision_threshold, color='y', linewidth=3, linestyle='dashed',
            label='Percentile-based Decision Threshold = {:0.3f}'.format(decision_threshold))

plt.legend(loc='upper left')

fig = plt.gcf()  # Get the current figure


pred_test = []
true_test = []
labels =[]


# calculate the reconstruction error for federated test data

# to not show the training information like ms/step
tf.keras.utils.disable_interactive_logging()

for batch in federated_test_prediction:
    for feature in batch:

        x_pred = trained_global_model.predict(feature['x'])
        x_pred = np.array(x_pred).reshape(x_pred.shape[0], x_pred.shape[1], x_pred.shape[2])
        pred_test.append(x_pred)
        true_test.append(np.array(feature['x']).reshape(feature['x'].shape[0], feature['x'].shape[1], feature['x'].shape[2]))
        labels.append(feature['y'])


test_true = []
test_true = [i for j in true_test for i in j]

test_pred = []
test_pred = [i for j in pred_test for i in j]

y_true = []
y_true = [i for j in labels for i in j]

test_true = np.array(test_true)
test_true = test_true.reshape(test_true.shape[0], test_true.shape[1], test_true.shape[2])

test_pred = np.array(test_pred)
test_pred = test_pred.reshape(test_pred.shape[0], test_pred.shape[1], test_pred.shape[2])

y_true=np.array(y_true)
#y_true=np.argmax(y_true, axis=-1)

# Compute reconstruction error
cross_ent_from_test_prediction = tf.keras.backend.binary_crossentropy(test_true, test_pred)
test_reconstruction_error = tf.reduce_sum(cross_ent_from_test_prediction, axis=[1,2]) #consolidate at each instance

#### Present the latent space of the FL-CVAE

encoder = trained_global_model.get_layer('encoder')

encoder.summary()


test_y_pred = np.array([1 if (x > decision_threshold) else 0 for x in test_reconstruction_error])


