import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig


# Split the dataframe into test and train data
def split_data(df):
    X = df["data"]
    y = df["label"]

    # Lists for proper splitting
    arrays_label = []
    X_train_lst = []
    X_test_lst = []
    y_train_lst = []
    y_test_lst = []    

    for label in np.unique(y):  
        index = np.where((y==label)) #return the index with that label
        
        # Index has two arrays, one wiht the positions and one filled with zeros
        # We add the positions to x and y
        x_l = X[index[0],:] 
        Y_l = y[index[0]]
        
        #Split the values 70/30 in a random mode for the values acquired in the previous step
        X_train_one_label, X_test_one_label, y_train_one_label, y_test_one_label = train_test_split( x_l, Y_l, train_size=0.7, test_size=0.3, shuffle=False)
        
        #Complete list with all the labels 70/30
        X_train_lst.append( X_train_one_label)
        X_test_lst.append( X_test_one_label)

        y_train_lst.append( y_train_one_label)
        y_test_lst.append( y_test_one_label)

    # Put in an array the lists for different classes
    X_train = np.vstack(X_train_lst)
    X_test = np.vstack(X_test_lst)

    y_train = np.vstack(y_train_lst)
    y_test = np.vstack(y_test_lst)

    #X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, test_size=0.2, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    return data


# Train the model, return the model
def train_model(data):
    reg_model = SVC(kernel='linear',
                    probability=True,
                    decision_function_shape='ovo')
    reg_model.fit(data["train"]["X"], data["train"]["y"])
    return reg_model


# Evaluate the metrics for the model
def get_model_metrics(model, data):
    preds = model.predict(data["test"]["X"])
    accuracy = accuracy_score(preds, data["test"]["y"])
    metrics = {"accuracy": accuracy}
    return metrics


def main():
    print("Running train.py")

    # Load the training data as dataframe
    data_dir = "data"
    data_file = os.path.join(data_dir, 'ID0018C09_dataset.mat')
    #train_df = pd.read_csv(data_file)
    train_df = loadmat(data_file)

    data = split_data(train_df)

    # Train the model
    model = train_model(data)

    # Log the metrics for the model
    metrics = get_model_metrics(model, data)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()





experiment = Experiment(workspace=ws, name='nemesis SVM')
config = ScriptRunConfig(source_directory='./src',
                            script='train.py',
                            compute_target='cpu-cluster')