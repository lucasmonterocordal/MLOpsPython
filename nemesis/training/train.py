"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""


import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Split the dataframe into test and train data
def split_data(df):
    X = df.drop('Y', axis=1).values
    y = df['Y'].values
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
    y_train = np.vstack(y_train_lst).flatten()    
    y_test = np.vstack(y_test_lst)
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
    # data_file = os.path.join(data_dir, 'diabetes.csv')
    data_file = os.path.join(data_dir, 'ID0018C09_dataset.csv')
    train_df = pd.read_csv(data_file)
    data = split_data(train_df)
    # Train the model
    model = train_model(data)
    # Log the metrics for the model
    metrics = get_model_metrics(model, data)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
