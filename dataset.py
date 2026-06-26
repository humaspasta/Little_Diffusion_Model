from sklearn.datasets import make_moons
import math
import numpy as np
import torch

class MakeData():

    
    def make_moons_data(self, num_points=100):
        '''
        Creates two moons data with normalized points centered around 0,0:
        num_points: the number of points in the datast
        '''
        X,y = make_moons(n_samples=num_points)

        print(X)

        print(X.shape)
        print(y.shape)
        x_coords = torch.tensor([val[0] for val in X])
        y_coords = torch.tensor([val[1] for val in X])
        x_max , x_min = x_coords.max() , x_coords.min()
        y_max , y_min = y_coords.max() , y_coords.min()
        
        x_mean = x_coords.mean()
        y_mean = y_coords.mean()

        #center so that sample mean is 0
        x_coords = x_coords - x_mean 
        y_mean = y_coords-y_mean

        #normalize each between zero and one using min-max normalization
        x_coords = (x_coords - x_min) / x_max - x_min
        y_coords = (y_coords - y_min) / y_max - y_min 

        

    

        return x_coords , y_coords





        
        
