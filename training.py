import torch
from model import basic_model
import random
import math
from torch.utils.data import TensorDataset, DataLoader
import os

def train_pattern_diffuse(x_data , y_data, model, optimizer, epochs=1000, noise_steps=100,batch_size=1 , name='model'):
    '''
    x_data: the set of points on the x-axis
    y_data: the set of corresponding points on the y_axis
    optimizer: The pytorch optimizer for training (I used Adam)
    epochs: number of epochs
    noise_steps: number of noisy steps you want to train with
    batch_size: number of batches for training
    name: the filename for the model you wish to save in ./weights 

    Returns: array of floats for loss
    '''
    assert len(x_data) == len(y_data) 

    data = torch.stack([x_data , y_data] , dim = 1) #stacked so that [(x_0 , y_0), (x_1,y_1), ... , (x_n,y_n)]^T for all n points

    dataset = TensorDataset(data)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_arr = []
    betas = torch.linspace(1e-4, 0.02, steps=noise_steps)
    alphas = 1 - betas

    alpha_bars = torch.cumprod(alphas, dim=0)

    saved_path = os.path.join('.' ,'weights', name)

    for i in range(epochs):
        for batch in loader:

            optimizer.zero_grad()
            
            points = batch[0]
            


            t = torch.randint(0, noise_steps, (len(points),)) # get time steps for noise ()

            a_bars = alpha_bars[t].unsqueeze(1) #

            s_epsilon= torch.randn(size=(len(points),2)) # sample the epsilons for each x,y (points , 2)

            inp_pair = torch.sqrt(a_bars) * points + torch.sqrt(1 - a_bars)*s_epsilon
            normalized_t = (t/noise_steps).unsqueeze(1)

            inp_data = torch.cat([inp_pair , normalized_t], dim=1)

            epsilons = model(x=inp_data)

            loss = torch.nn.functional.mse_loss(epsilons, s_epsilon)
            loss_arr.append(loss.item())
            
            print(f"\r Epoch {i} loss: {loss.item()}",end="",flush=True)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), saved_path)

    return loss_arr
    

def recreate_pattern_dataset(model , T , n_samples=200):

    '''
    This is the sampling step for any pattern. The sampling step is the step used to recreate your pattern.

    model: the trained model
    T: The number of time steps used for training
    n_samples: the number of samples you want in your batch size (generation is done on a batched basis)
    '''
    with torch.no_grad():
        x_t = torch.randn(n_samples , 2)
        
        betas = torch.linspace(1e-4, 0.02, steps=T)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
    
        for t in range(T-1, -1, -1):
            z = torch.randn(n_samples, 2) if t >= 1 else torch.zeros(n_samples, 2)

            t_normal = torch.full((n_samples , 1) , t/T)
            inp = torch.cat([x_t , t_normal] , dim = 1)

            model_vals = model(x=inp)


            x_t = (1/math.sqrt(alphas[t])) * (x_t - ((1 - alphas[t]) / math.sqrt(1 - alpha_bars[t])) * model_vals) + math.sqrt(betas[t]) * z
            if t % 100 == 0:
                print(f'\r Step {t}' , end='\n', flush=True)
    return x_t
            










            