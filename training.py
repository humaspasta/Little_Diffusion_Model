import torch
from model import basic_model
import random
import math


def train_two_moons_diffuse(x_data , y_data, model, optimizer, epochs=10000, noise_steps=100):
    assert len(x_data) == len(y_data) 
    betas = torch.linspace(1e-4, 0.02, steps=noise_steps)
    alphas = 1 - betas

    alpha_bars = torch.cumprod(alphas, dim=0)

    for i in range(epochs):
        optimizer.zero_grad()
       
        t = random.randint(0, noise_steps - 1)
        
        idx = random.randint(0, len(x_data) - 1)
        x_0 , y_0 = x_data[idx] , y_data[idx]

        s_epsilon= torch.randn(size=(2,))

        inp_x = torch.sqrt(alpha_bars[t])*x_0 + torch.sqrt(1-alpha_bars[t])*s_epsilon[0]
        inp_y = torch.sqrt(alpha_bars[t])*y_0 + torch.sqrt(1-alpha_bars[t])*s_epsilon[1]

        epsilons = model(x=torch.tensor(data=[inp_x , inp_y, t] ,dtype=torch.float32))

        print(epsilons)
        loss = torch.nn.functional.mse_loss(epsilons, s_epsilon)
        
        print(f"\r Epoch {i} loss: {loss.item()}",end="",flush=True)
        loss.backward()
        optimizer.step()
    

def recreate_two_moons_dataset(model , T):
    '''
    This is the sampling step for two moons
    '''
    x_t = torch.randn(())
    y_t = torch.randn(())

    betas = torch.linspace(1e-4, 0.02, steps=T)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    for t in range(T-1, -1, -1):
        z = None 
        if t >= 1:
            z = torch.randn(2)
        else:
            z = torch.zeros(2)
        model_vals = model(x=torch.tensor([x_t , y_t , t] , dtype=torch.float32))

        in_x = ((1-alphas[t]) / (math.sqrt(1 - alpha_bars[t]))) * model_vals[0]
        in_y = ((1-alphas[t]) / (math.sqrt(1 - alpha_bars[t]))) * model_vals[1]

        x_t= 1/math.sqrt(alphas[t]) * (x_t - in_x) + math.sqrt(betas[t]) * z[0]
        y_t = 1/math.sqrt(alphas[t]) * (y_t - in_y) + math.sqrt(betas[t]) * z[1]

    return x_t , y_t
            










            