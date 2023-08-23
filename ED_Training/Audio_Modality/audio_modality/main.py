import os
import torch
import torch.nn as nn
import pandas as pd
from conf import MAIN_FOLDER


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
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.output(x)
        return x


if __name__ == '__main__':
    dataset_path = os.path.join(MAIN_FOLDER, 'datasets', 'ravdess_dataset_features.csv')
    dataset = pd.read_csv(dataset_path)
    train, test = torch.utils
