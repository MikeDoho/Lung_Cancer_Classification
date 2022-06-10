import os
import ast
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from lib.utils.logger import log


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
      E.g. for use with categorical_crossentropy.
      Args:
          y: class vector to be converted into a matrix
              (integers from 0 to num_classes).
          num_classes: total number of classes. If `None`, this would be inferred
            as the (largest number in `y`) + 1.
          dtype: The data type expected by the input. Default: `'float32'`.
      Returns:
          A binary matrix representation of the input. The classes axis is placed
          last.
      Example:
      >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
      >>> a = tf.constant(a, shape=[4, 4])
      >>> print(a)
      tf.Tensor(
        [[1. 0. 0. 0.]
         [0. 1. 0. 0.]
         [0. 0. 1. 0.]
         [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)
      >>> b = tf.constant([.9, .04, .03, .03,
      ...                  .3, .45, .15, .13,
      ...                  .04, .01, .94, .05,
      ...                  .12, .21, .5, .17],
      ...                 shape=[4, 4])
      >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
      >>> print(np.around(loss, 5))
      [0.10536 0.82807 0.1011  1.77196]
      >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
      >>> print(np.around(loss, 5))
      [0. 0. 0. 0.]
      Raises:
          Value Error: If input contains string value
      """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


#  LOAD FUNCTIONS

class clinical_data:

    def __init__(self, args, train_mrn_list=[], val_mrn_list=[], test_mrn_list=[]):

        self.args = args
        self.train_mrn_list = [float(x) for x in train_mrn_list]
        self.val_mrn_list = [float(x) for x in val_mrn_list]
        self.test_mrn_list = [float(x) for x in test_mrn_list]

    def load_val_data_batch(self,
                            batch_size=32):

        # required when using hyperopt
        batch_size = int(batch_size)

        # self.args.selected_clinical_features = ast.literal_eval(self.args.selected_clinical_features)

        # # Load MRN files
        # Obtaining MRNs for directories as the directories are changing when using the data_setup function in
        # the model .py file
        # val_list = [x.split('_')[0] for x in os.listdir(dir_dict['dir_val'])]
        # train_list = [x.split('_')[0] for x in os.listdir(dir_dict['dir_train']) if
        #               not x.split('_')[-1].strip('.npy').isnumeric()]

        # Setting values for images and outcomes
        num_classes = self.args.n_classes

        ## Addition for Clinical Data
        # loading clinical data
        # os.chdir(self.args.clinical_data_path)
        clinical_data = pd.read_csv(os.path.join(self.args.clinical_data_path, self.args.clinical_data_filename))

        # log.info(ast.literal_eval(self.args.selected_clinical_features))

        X_train = clinical_data[clinical_data['MRN'].isin(self.train_mrn_list)].drop(
            columns=['new_outcome_ft_weight_loss'])
        y_train = clinical_data[clinical_data['MRN'].isin(self.train_mrn_list)]['new_outcome_ft_weight_loss']

        keep_val_mrn = clinical_data[clinical_data['MRN'].isin(self.val_mrn_list)]['MRN'].tolist()
        X_val = clinical_data[clinical_data['MRN'].isin(self.val_mrn_list)].drop(columns=['new_outcome_ft_weight_loss'])

        # Added to address new imputing
        col_names = X_train.columns.tolist()

        # Set up for data imputing and scaling
        knn_imputer = KNNImputer(n_neighbors=5)
        impute_fit = knn_imputer.fit(X=X_train, y=y_train)
        scaler = MinMaxScaler(feature_range=(0, 1))

        # X train
        impute_x_train = impute_fit.transform(X=X_train)
        impute_x_train = pd.DataFrame(impute_x_train, columns=col_names)
        # removing selected features for now
        impute_x_train = np.array(impute_x_train[ast.literal_eval(self.args.selected_clinical_features)])
        impute_x_train = scaler.fit_transform(impute_x_train)
        impute_x_train = pd.DataFrame(impute_x_train, columns=ast.literal_eval(self.args.selected_clinical_features))

        # X val
        impute_x_val = impute_fit.transform(X=X_val)
        impute_x_val = pd.DataFrame(impute_x_val, columns=col_names)
        # removing selected features for now (will have to add after)
        impute_x_val = np.array(impute_x_val[ast.literal_eval(self.args.selected_clinical_features)])
        impute_x_val = scaler.transform(impute_x_val)
        impute_x_val = pd.DataFrame(impute_x_val, columns=ast.literal_eval(self.args.selected_clinical_features))
        impute_x_val['MRN'] = keep_val_mrn

        # need to return to clinical data
        clinical_data = impute_x_val.reset_index(drop=True)

        # Creating list val inputs
        val_x_choices = self.val_mrn_list
        val_x_choices = list(map(int, val_x_choices))
        val_x_choices = list(map(str, val_x_choices))

        # Filter validation input files
        filter_val = val_x_choices

        # log.info('val')

        while True:
            count = 0
            # Setting up input information
            y_val = np.zeros((batch_size, num_classes))
            x_clinical_val = np.zeros((batch_size, np.shape(clinical_data)[1]))

            for file in np.random.choice(filter_val, batch_size, replace=False):
                # data label as an integer
                data_label = np.load(os.path.join(self.args.val_label_path, file + '_new_outcome.npy'))
                y_val[count, :] = to_categorical(data_label, num_classes=num_classes)

                # Loading/storing clinical data # will have to check file.split...needs to be integer or str
                index = clinical_data.loc[clinical_data['MRN'] == float(file)].index.tolist()[0]
                pt_data = clinical_data.iloc[index]
                pt_data = pt_data.drop(columns=['MRN']).to_numpy()
                x_clinical_val[count, ...] = pt_data

                count += 1

            return x_clinical_val, y_val

    def load_test_data_batch(self,
                             batch_size=32):
        """
        :param batch_size: number of inputs to load
        :param dir_dict: dictionary
        :return: x_test, y_test (npy files)
        """

        # required when using hyperopt
        batch_size = int(batch_size)

        # Loading MRNs that are obtained from directories as directories are changing with data_setup function in model .py file
        # test_list = [x.split('_')[0] for x in os.listdir(dir_dict['dir_test'])]
        # train_list = [x.split('_')[0] for x in os.listdir(dir_dict['dir_train']) if
        #               not x.split('_')[-1].strip('.npy').isnumeric()]

        # Setting values for images and outcomes
        num_classes = self.args.n_classes

        ## Addition for Clinical Data
        # loading clinical data
        # os.chdir(self.args.clinical_data_path)
        clinical_data = pd.read_csv(os.path.join(self.args.clinical_data_path, self.args.clinical_data_filename))

        keep_test_mrn = clinical_data[clinical_data['MRN'].isin(self.test_mrn_list)]['MRN'].tolist()
        X_train = clinical_data[clinical_data['MRN'].isin(self.train_mrn_list)].drop(
            columns=['new_outcome_ft_weight_loss', 'MRN'])
        y_train = clinical_data[clinical_data['MRN'].isin(self.train_mrn_list)]['new_outcome_ft_weight_loss']

        X_test = clinical_data[clinical_data['MRN'].isin(self.test_mrn_list)].drop(
            columns=['new_outcome_ft_weight_loss', 'MRN'])

        # Added to address new imputing
        col_names = X_train.columns.tolist()

        # Set up for data imputing and scaling
        knn_imputer = KNNImputer(n_neighbors=5)
        impute_fit = knn_imputer.fit(X=X_train, y=y_train)
        scaler = MinMaxScaler(feature_range=(0, 1))

        # X train
        impute_x_train = impute_fit.transform(X=X_train)
        impute_x_train = pd.DataFrame(impute_x_train, columns=col_names)

        # removing selected features for now
        impute_x_train = np.array(impute_x_train[ast.literal_eval(self.args.selected_clinical_features)])
        impute_x_train = scaler.fit_transform(impute_x_train)
        impute_x_train = pd.DataFrame(impute_x_train, columns=ast.literal_eval(self.args.selected_clinical_features))

        # # X test
        impute_x_test = impute_fit.transform(X=X_test)
        impute_x_test = pd.DataFrame(impute_x_test, columns=col_names)
        # removing selected features for now
        impute_x_test = np.array(impute_x_test[ast.literal_eval(self.args.selected_clinical_features)])
        impute_x_test = scaler.transform(impute_x_test)
        impute_x_test = pd.DataFrame(impute_x_test, columns=ast.literal_eval(self.args.selected_clinical_features))

        # need to return MRN because it was getting normalized above
        impute_x_test['MRN'] = keep_test_mrn

        # need to return to clinical data
        clinical_data = impute_x_test.reset_index(drop=True)

        # Creating list test inputs
        test_x_choices = self.test_mrn_list
        test_x_choices = list(map(int, test_x_choices))
        test_x_choices = list(map(str, test_x_choices))

        filter_test = test_x_choices

        # log.info('test')

        while True:
            count = 0
            # subtracting 2 because we will be excluding MRN later
            x_clinical_test = np.zeros((batch_size, np.shape(clinical_data)[1]))

            # Setting up input information
            y_test = np.zeros((batch_size, num_classes))

            for file in np.random.choice(filter_test, batch_size, replace=False):
                # data label as an integer
                # log.info(file)
                data_label = np.load(os.path.join(self.args.test_label_path, file + '_new_outcome.npy'))
                y_test[count, :] = to_categorical(data_label, num_classes=num_classes)

                # Loading/storing clinical data
                index = clinical_data.loc[clinical_data['MRN'] == float(file)].index.tolist()[0]
                pt_data = clinical_data.iloc[index]
                pt_data = pt_data.drop(columns=['MRN']).to_numpy()
                x_clinical_test[count, ...] = pt_data

                count += 1

            return x_clinical_test, y_test

    def load_train_data_batch(self,
                              batch_size=32):
        """
        :param oversample_value: creates additional positive samples to draw from to increase positive sampling frequency
        :param batch_size: number of inputs to load
        :param dir_dict: dictionary
        :return: x_test, y_test (npy files)
        """

        # required when using hyperopt
        batch_size = int(batch_size)

        # Setting values for images and outcomes
        num_classes = self.args.n_classes

        ## Addition for Clinical Data
        # loading clinical data
        # os.chdir(self.args.clinical_data_path)
        clinical_data = pd.read_csv(os.path.join(self.args.clinical_data_path, self.args.clinical_data_filename))
        # print(clinical_data)

        # log.info(self.train_mrn_list)

        # # Load MRN files
        # Obtaining MRNs for directories as the directories are changing when using the data_setup function in
        # the model .py file

        keep_train_mrn = clinical_data[clinical_data['MRN'].isin(self.train_mrn_list)]['MRN'].tolist()

        X_train = clinical_data[clinical_data['MRN'].isin(self.train_mrn_list)].drop(
            columns=['new_outcome_ft_weight_loss', 'MRN'])
        y_train = clinical_data[clinical_data['MRN'].isin(self.train_mrn_list)]['new_outcome_ft_weight_loss']

        # Added to address new imputing
        col_names = X_train.columns.tolist()

        # Set up for data imputing and scaling
        knn_imputer = KNNImputer(n_neighbors=5)
        impute_fit = knn_imputer.fit(X=X_train, y=y_train)
        scaler = MinMaxScaler(feature_range=(0, 1))

        # X train
        impute_x_train = impute_fit.transform(X=X_train)
        impute_x_train = pd.DataFrame(impute_x_train, columns=col_names)
        # removing selected features for now
        impute_x_train = np.array(impute_x_train[ast.literal_eval(self.args.selected_clinical_features)])
        impute_x_train = scaler.fit_transform(impute_x_train)
        impute_x_train = pd.DataFrame(impute_x_train, columns=ast.literal_eval(self.args.selected_clinical_features))

        # need to return MRN because it was getting normalized above
        impute_x_train['MRN'] = keep_train_mrn

        # need to return to clinical data
        clinical_data = impute_x_train.reset_index(drop=True)

        filter_train = self.train_mrn_list
        filter_train = list(map(int, filter_train))
        filter_train = list(map(str, filter_train))

        log.info('train')

        while True:
            count = 0
            # subtracting 2 because we will be excluding MRN later
            x_clinical_train = np.zeros((batch_size, np.shape(clinical_data)[1]))
            # Setting up input information
            y_train = np.zeros((batch_size, num_classes))

            for file in np.random.choice(filter_train, batch_size, replace=False):
                # data label as an integer
                # log.info(file)
                data_label = np.load(os.path.join(self.args.train_label_path, file + '_new_outcome.npy'))
                y_train[count, :] = to_categorical(data_label, num_classes=num_classes)

                # Loading/storing clinical data
                index = clinical_data.loc[clinical_data['MRN'] == float(file)].index.tolist()[0]
                pt_data = clinical_data.iloc[index]
                pt_data = pt_data.drop(columns=['MRN']).to_numpy()
                x_clinical_train[count, ...] = pt_data

                count += 1

            return x_clinical_train, y_train
