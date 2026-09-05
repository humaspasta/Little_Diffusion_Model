from sklearn.datasets import make_moons , make_blobs , make_swiss_roll, make_circles
import math
import numpy as np
import torch


class MakeData():

    
    def make_normal_data(self, n_points=100 , type='moons'):
        '''
        Creates two moons data with normalized points centered around 0,0:
        num_points: the number of points in the datast
        type: moons, blobs, swiss_rolls, circles
        '''
        patterns = dict()

        patterns['moons'] = make_moons 
        patterns['blobs'] = make_blobs
        patterns['swiss roles'] = make_swiss_roll 
        patterns['circles'] = make_circles 


        

        if type not in patterns.keys():
            raise ValueError('Pattern type not in patterns list. Please choose a pattern that exists.')
        X , y = patterns[type](n_points)

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
    



        





        
        
