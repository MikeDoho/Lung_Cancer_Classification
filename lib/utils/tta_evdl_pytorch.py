import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# Evidential ML
from evdl.pytorch_class_uncertainty.losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, \
    relu_evidence


# def enable_dropout(model):
#     """ Function to enable the dropout layers during test-time """
#     for m in model.modules():
#         if m.__class__.__name__.startswith('Dropout'):
#             m.train()

### FOR THIS TO WORK THE DATA LOADER NEEDS TO DO AUGMENTATION ON EVERY PULL

def get_tta_predictions(data_loader,
                        forward_passes,
                        model,
                        n_classes,
                        n_samples):
    """ Function to get tta and uncertainty estimates
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

    tta_predictions = np.empty((0, n_samples, n_classes))
    softmax = nn.Softmax(dim=1)

    # single_prediction_storage = []
    label_storage = []
    mrn_storage = []

    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        model.eval()
        for _, (image, label, mrn) in enumerate(data_loader):
            image = image.to(torch.device('cuda'))
            with torch.no_grad():
                outputs = model(image)

                # preds is either 0 or 1. so far there are no probabilities output
                _, preds = torch.max(outputs, 1)
                evidence = relu_evidence(outputs)
                alpha = evidence + 1

                # u is the metric for uncertainty where 1 is total uncertainty (check paper again)
                u = n_classes / torch.sum(alpha, dim=1, keepdim=True)

                ## Mike is adding calculations per his interpretation of paper
                S = n_classes / u
                class_prob_pred = alpha / S

                outputs = class_prob_pred

                if i == 0:
                    # to store the MRN and the label for each patient (only look at first pass, i == 0)
                    label_storage.append(label.cpu().numpy()[0])
                    mrn_storage.append(mrn[0])

            predictions = np.vstack((predictions, outputs.cpu().numpy()))

        tta_predictions = np.vstack((tta_predictions,
                                         predictions[np.newaxis, :, :]))
        # dropout predictions - shape (forward_passes, n_samples, n_classes)

    #
    # print(dropout_predictions)

    # Calculating mean across multiple MCD forward passes
    mean = np.mean(tta_predictions, axis=0)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes
    variance = np.var(tta_predictions, axis=0)  # shape (n_samples, n_classes)

    epsilon = 1e-12
    # Calculating entropy across multiple MCD forward passes
    entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

    df = pd.DataFrame()
    df['mrn'] = mrn_storage
    df['label'] = label_storage
    df['pred_0'] = mean[:, 0]
    df['pred_1'] = mean[:, 1]
    df['var_0'] = variance[:, 0]
    df['var_1'] = variance[:, 1]
    df['mean_entropy'] = entropy

    return df
