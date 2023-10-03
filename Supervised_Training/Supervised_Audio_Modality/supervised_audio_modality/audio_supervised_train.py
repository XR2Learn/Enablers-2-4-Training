import os
import numpy as np
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Tensor
import torch.nn as nn
from torch.optim import SGD

from supervised_audio_modality.conf import MAIN_FOLDER


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        self.X = None
        self.y = None
        self.load_x_y_from_csv(path)
        # load the csv file as a dataframe
        # df = read_csv(path)
        # # store the inputs and outputs
        # self.X = df['Features Path']
        # self.y = df['Emotion']
        # # label encode target and ensure the values are floats
        # self.y = LabelEncoder().fit_transform(self.y)
        # self.y = self.y.astype('float32')
        # self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        example_features = self.get_example_features(idx)
        label = self.y[idx]
        return [example_features, label]

    def get_example_features(self, idx):
        # Loading the features from file
        example_features = np.load(self.X[idx], allow_pickle=True, fix_imports=False)
        example_features = example_features.squeeze(axis=1)
        return example_features

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

    def load_x_y_from_csv(self, path):
        # load the csv file as a dataframe
        df = read_csv(path)
        # store the inputs and outputs
        self.X = df['Features Path']
        self.y = df['Emotion']
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))
        print(self.y)


class MLPClassifier(nn.Module):
    """ MLP for classification
    """

    def __init__(self, in_size, out_size, hidden=[256, 128]):
        super(MLPClassifier, self).__init__()
        self.name = 'MLP'
        self.relu = nn.ReLU()
        # TODO: make dropout optional with an argument
        self.linear1 = nn.Sequential(
            nn.Linear(in_size, hidden[0]),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2)
        )
        self.output = nn.Linear(hidden[1], out_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.output(x)
        return x


# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=20, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl


# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
        if epoch == 5:
            break
        print(f'End of Epoch {epoch}, loss: {loss}')


# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc


# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat


if __name__ == '__main__':
    # prepare the data
    path = os.path.join(MAIN_FOLDER, 'datasets', 'ravdess_dataset_features.csv')
    train_dl, test_dl = prepare_data(path)
    print(f"Size train set: {len(train_dl.dataset)}, \nSize test set: {len(test_dl.dataset)}")
    # define the network
    model = MLPClassifier(4137984, 1)
    # train the model
    train_model(train_dl, model)
    # evaluate the model
    acc = evaluate_model(test_dl, model)
    print('Accuracy: %.3f' % acc)
