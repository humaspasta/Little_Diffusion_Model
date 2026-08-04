from dataset import MakeData
import matplotlib.pyplot as plt
import training
from model import basic_model
import torch
import torch.optim as optim


datamaker = MakeData()
N_POINTS = 200
x,y = datamaker.make_moons_data(N_POINTS)
noisy_x,noisy_y = datamaker.generate_noisy_moons(noise=0.9)

curr_model = basic_model()
optimizer = optim.Adam(curr_model.parameters() , lr=0.001)
training.train_two_moons_diffuse(x ,y, model=curr_model, optimizer=optimizer, epochs=100000)
samples_x , samples_y = [] , []

for _ in range(N_POINTS):
    x, y = training.recreate_two_moons_dataset(curr_model, T=1000)
    samples_x.append(x.item())
    samples_y.append(y.item())





torch.save(curr_model.state_dict(), 'model_weights.pth')

plt.scatter(x.detach().numpy() , y.detach().numpy() , label='signal')
plt.scatter(noisy_x,noisy_y, label='noise')
plt.scatter(samples_x , samples_y, label='learned')
plt.legend()
plt.show()

