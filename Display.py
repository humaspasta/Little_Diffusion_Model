import os
import torch 
import numpy 
from model import basic_model
import training 
import matplotlib.pyplot as plt
from dataset import MakeData

'''
This file is meant to experiment with the already trained model
'''

patterns = ['blobs']
datamaker = MakeData()
for pattern in patterns:
    regular_x , regular_y = datamaker.make_normal_data(n_points = 200, type=pattern)
    figure_path = os.path.join('.' , 'Figures', f'{pattern}.png')
    pattern_path = os.path.join('.' , 'weights', pattern)
    model = basic_model()
    state_dict_trained = torch.load(pattern_path)
    model.load_state_dict(state_dict_trained)
    data = training.recreate_pattern_dataset(model , T=2000)
    x_data = data[ :, 0].numpy()
    y_data = data[ :,  1].numpy()

    plt.scatter(regular_x , regular_y, label='original')
    plt.scatter(x_data , y_data, label='learned')
   

    plt.title(f'Trained {pattern}')
    plt.legend()
    plt.savefig(figure_path)
    plt.show()






