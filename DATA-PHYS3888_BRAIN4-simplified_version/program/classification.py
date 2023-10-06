# Classifier methods called by stream.py 
from tsfresh.feature_extraction import extract_features, MinimalFCParameters, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy.stats import kurtosis
from sklearn.impute import SimpleImputer
import pickle
import pandas as pd
import numpy as np
import numpy as np

def load_classifier(file_name):
    """
    A method to load classifier given .pkl files
    parameter file_name: the relative path of .pkl file
    return: classifier
    """
    with open(file_name, 'rb') as file:
        return pickle.load(file)

def load_all_classifiers(filenames):
    """
    A method used to load all the classifiers
    parameter filenames: a list containing all file names
    return: a list of classifier
    """
    classifiers = {}
    for filename in filenames:
        classifier_name = filename[:-4].replace("_", " ")
        classifiers[classifier_name] = load_classifier(filename)
        print(f"\n{classifier_name} classifier loaded.")
    return classifiers

def predict(signal, classifier):
    custom_fc_parameters = {
    'mean': None,
    'median': None,
    'kurtosis': None,
    'mean_abs_change': None,
    'quantile': [{'q': 0.25}, {'q': 0.75}],
    'count_below_mean': None,
    'cid_ce': [{'normalize': True}],
    }
    df = pd.DataFrame({'id': 0, 'time': np.arange(len(signal)), 'value': signal.astype(float)})
    extracted_features = extract_features(df, column_id='id', column_sort='time', default_fc_parameters=custom_fc_parameters)
    impute(extracted_features)
    predicted_class = classifier.predict(extracted_features)
    return predicted_class

def extract_features_manual(data):
    """
    manually extract all the selected features
    """
    data = np.array(data).astype(float)
    sum_values = np.sum(data)
    median = np.median(data)
    mean = np.mean(data)
    length = len(data)
    std_dev = np.std(data)
    variance = np.var(data)
    root_mean_square = np.sqrt(np.mean(data**2))
    maximum = np.max(data)
    absolute_maximum = np.max(np.abs(data))
    minimum = np.min(data)

    features = [
        sum_values,
        median,
        mean,
        length,
        std_dev,
        variance,
        root_mean_square,
        maximum,
        absolute_maximum,
        minimum,
    ]
    return features

def predict_label(data, classifier):
    """
    parameter data: signals
    classifier: chosen best classifier
    return: predicted label of the signals
    """
    data_real = np.real(data)  # Extract the real part of the complex numbers
    features = extract_features_manual(data_real)
    features = np.array(features).reshape(1, -1)  # Reshape the features to a 2D array
    label = classifier.predict(features) # predict the label using our classifier
    return label

def deleteHeadAndTail(ls_label):
    """
    This method is used to delete the head and the tail of the event list
    """
    return ls_label[1:-1] if len(ls_label)>3 else ls_label