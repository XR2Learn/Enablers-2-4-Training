import numpy as np

def create_dummy_data(n_channels=1,length=2500,size=1000,n_labels=3):
    dummy_data = np.zeros((size,n_channels,length))
    dummy_labels = np.random.random_integers(low=0,high=n_labels,size=size)

    return dummy_data,dummy_labels