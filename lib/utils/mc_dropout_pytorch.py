#https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch
# Author: Iswariya Manivannan

import sys

import numpy as np
import pandas as pd
import copy

import torch
import torch.nn as nn
from lib.utils.logger import log


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    # print('ENABLE DROPOUT; SEE LAYERS BELOW')
    # log.info('ENABLE DROPOUT; SEE LAYERS BELOW')
    for m in model.modules():
        # if m.__class__.__name__.startswith('Dropout') or m.__class__.__name__.startswith('Dropout3d'):
        if m.__class__.__name__.startswith('Dropout'):
            # print(m.__class__.__name__)
            # log.info(m.__class__.__name__)
            m.train()
    # print('***************************\n')


def get_monte_carlo_predictions(data_loader,
                                forward_passes,
                                model,
                                n_classes,
                                n_samples):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """

    single_prediction_model = copy.deepcopy(model)
    dropout_predictions = np.empty((0, n_samples, n_classes))
    # df = pd.DataFrame()
    single_prediction_storage = []
    label_storage = []
    mrn_storage = []

    softmax = nn.Softmax(dim=1)

    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        model.eval()
        single_prediction_model.eval()
        # print('can I see this')
        # log.info('can I see this')
        enable_dropout(model)
        for j, (image, label, mrn) in enumerate(data_loader):
            # print(j)
            # mrn = list(input_tuple[2])
            # print(mrn_str)
            # input_tuple = input_tuple[0:2]

            image = image.to(torch.device('cuda'))
            with torch.no_grad():
                output = model(image)
                output = softmax(output)  # shape (n_samples, n_classes)

                # single prediction model
                if i == 0:
                    # print('prediction: ', softmax(single_prediction_model(image)).cpu().numpy()[0][1])
                    class_1_pred = softmax(single_prediction_model(image)).cpu().numpy()[0][1]
                    class_0_pred = softmax(single_prediction_model(image)).cpu().numpy()[0][0]
                    assert_value = abs((class_0_pred + class_1_pred)-1)
                    assert assert_value <= 0.001, f"[{class_0_pred}, {class_1_pred}], {assert_value}"

                    single_prediction_storage.append(class_1_pred)
                    # print(label.cpu().numpy()[0])
                    # print(mrn[0])
                    label_storage.append(label.cpu().numpy()[0])
                    mrn_storage.append(mrn[0])

            predictions = np.vstack((predictions, output.cpu().numpy()))

        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
        # dropout predictions - shape (forward_passes, n_samples, n_classes)

    #
    # print(dropout_predictions)

    # Calculating mean across multiple MCD forward passes
    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes
    variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    epsilon = 1e-12
    # Calculating entropy across multiple MCD forward passes
    entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

    df = pd.DataFrame()
    print('\n\nNew Array Lengths:')
    print(len(mrn_storage), len(label_storage), len(single_prediction_storage))
    print(mrn_storage)
    print(label_storage)
    print(single_prediction_storage)
    df['mrn'] = mrn_storage
    df['label'] = label_storage
    df['single_pred_1'] = single_prediction_storage
    df['pred_0'] = mean[:, 0]
    df['pred_1'] = mean[:, 1]
    df['var_0'] = variance[:, 0]
    df['var_1'] = variance[:, 1]
    df['mean_entropy'] = entropy

    # print(df)

    # # Calculating mutual information across multiple MCD forward passes
    # mutual_info = entropy - np.mean(np.sum(-dropout_predictions * np.log(dropout_predictions + epsilon),
    #                                        axis=-1), axis=0)  # shape (n_samples,)

    return df




