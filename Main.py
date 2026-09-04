from dataset import MakeData
import matplotlib.pyplot as plt
import training
from model import basic_model
import torch
import torch.optim as optim


datamaker = MakeData()
N_POINTS = 10000
x,y = datamaker.make_moons_data(N_POINTS)

x = (x - x.min()) / (x.max() - x.min()) * 2 - 1
y = (y - y.min()) / (y.max() - y.min()) * 2 - 1

curr_model = basic_model()
optimizer = optim.Adam(curr_model.parameters() , lr=0.001)

loss = training.train_two_moons_diffuse(x ,y, model=curr_model, optimizer=optimizer, epochs=1000, noise_steps=1000, batch_size=256)



res = training.recreate_two_moons_dataset(curr_model, T=1000)

res_x = res[:, 0].numpy()
res_y = res[:, 1].numpy()  

torch.save(curr_model.state_dict(), 'model_weights.pth')
fig , ax = plt.subplots(2,1)
ax[0].scatter(x.detach().numpy() , y.detach().numpy() , label='signal')

ax[0].scatter(res_x , res_y, label='learned')
ax[0].title('')
ax[0].legend()
ax[1].plot(loss)


plt.tight_layout()

plt.show()

