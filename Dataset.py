import torch as torch
import numpy as np

features_length=5*24
delay=24

class PRSA_dataset(torch.utils.data.Dataset):
    def __init__(self,dataframe):
        self.data=dataframe
    def __getitem__(self,index):
        the_data=(self.data.iloc[index: index+features_length+delay]).values
        feature=the_data[: delay,:]
        label=the_data[-1,0]

        return feature.astype(np.float32),label.astype(np.float32)
    def __len__(self):
        return len(self.data)-features_length-delay