# Python Modules
import os
import random

import numpy as np
import ast
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from interpret.glassbox import ExplainableBoostingClassifier


class clinical_model:
    """
    Will need to load training data outside of clinical model class
    """

    def __init__(self, args, X=None, y=None, model_type='lr'):

        self.args = args
        self.X = X
        self.y = y
        self.model_type = model_type

    def create_model(self):
        print('Loading clinical data for model training')
        # x_train_clinic, y_train_clinic = load_train_data_batch(batch_size=len(self.train_mrn_list),
        #                                                        args=args)

        print('Training data')
        print(np.shape(self.X))

        # Separate clinical model
        if self.model_type.lower() == 'lr':
            clinical_model_ = LogisticRegression(C=1, penalty='l1', solver='liblinear',
                                                 random_state=self.args.manual_seed)

        elif self.model_type.lower() == 'svm':
            clinical_model_ = svm.SVC(C=1, kernel='sigmoid', gamma='auto',
                                      random_state=self.args.manual_seed, probability=True)
        elif self.model_type.lower() == 'ebm':

            clinical_model_ = ExplainableBoostingClassifier(n_jobs=-1,
                                                            feature_names=ast.literal_eval(
                                                                self.args.selected_clinical_features),
                                                            random_state=self.args.manual_seed)
        elif self.model_type == 'mlp':
            clinical_model_ = MLPClassifier(hidden_layer_sizes=(12, 8, 6),
                                            alpha=0.0035,
                                            learning_rate_init=0.001,
                                            random_state=self.args.manual_seed+self.args.true_cv_count)
        else:
            clinical_model_ = None
            print('Need to select either lr or svm')

        print('Fitting model...')
        if not self.model_type.lower() == 'ebm':
            clinical_model_fit = clinical_model_.fit(self.X, np.argmax(self.y, axis=-1))
        else:
            # doing this specifically for ebm
            # print('ebm prior to fit')
            # print(np.shape(self.X))
            # print(self.X)
            # print('\n\n\n')
            clinical_model_fit = clinical_model_.fit(self.X, np.argmax(self.y, axis=-1))
        print('Model fit')

        return clinical_model_fit
