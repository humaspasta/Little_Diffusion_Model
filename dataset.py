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
        normalized_data = self.normalize(X=X)
        return normalized_data



    def generate_noisy_moons(self , n_points=100 , T=50, noise=0.2):
        X, _ = make_moons(n_samples=n_points , noise=noise)
        res = self.normalize(X = X)
        return res
    
    def normalize(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        x_coords = X[:, 0]
        y_coords = X[:, 1]

        # min-max normalize each dimension to [-1, 1]
        x_coords = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) * 2 - 1
        y_coords = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) * 2 - 1

        return x_coords, y_coords


        





        
        
