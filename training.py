import torch
import torch.nn as nn
from model import basic_model
import random


def train_two_moons_diffuse(x_data , y_data, model, optimizer, epochs=100, noise_steps=100):
    noise_steps = []
    assert len(x_data) == len(y_data) 
    for i in epochs:
        betas_x = torch.randn(noise_steps)
        betas_y = torch.randn(noise_steps)

        t = torch.randint(1).detatch().item()
        
        idx = random.randint(0, len(x_data) - 1)
        x_0 , y_0 = x_data[idx] , y_data[idx]

        s_epsilon= torch.randn(2).detach().item()
        
        alpha = 1 - betas_x[t]

        inp_x = torch.sqrt(alpha)*x_0 + torch.sqrt(1-alpha)*s_epsilon[0]
        inp_y = torch.sqrt(alpha)*y_0 + torch.sqrt(1-alpha)*s_epsilon[1]


        epsilons = model(inp_x , inp_y, t)
        loss = torch.norm(s_epsilon - epsilons)

        loss.backward()
        optimizer.step()





            




    

def basic_reverse_chain(output_points):
    pass 