from dataset import MakeData
import matplotlib.pyplot as plt
import training
from model import basic_model
import torch
import torch.optim as optim


datamaker = MakeData()

data = [] 

patterns = ['swiss roles']

N_POINTS = 10000
# x,y = datamaker.make_moons_data(N_POINTS)

curr_model = basic_model()
optimizer = optim.Adam(curr_model.parameters() , lr=0.001)


for pattern in patterns:
    print()
    print(f'pattern : {pattern}')
    x,y = datamaker.make_normal_data(N_POINTS, type=pattern)
    training.train_pattern_diffuse(x ,y, model=curr_model, optimizer=optimizer, epochs=100000, noise_steps=2000, batch_size=256, name=pattern)

    


# res = training.recreate_two_moons_dataset(curr_model, T=1000)

# res_x = res[:, 0].numpy()
# res_y = res[:, 1].numpy()  

# torch.save(curr_model.state_dict(), 'model_weights.pth')
# fig , ax = plt.subplots(2,1)
# ax[0].scatter(x.detach().numpy() , y.detach().numpy() , label='signal')

# ax[0].scatter(res_x , res_y, label='learned')
# ax[0].title('')
# ax[0].legend()
# ax[1].plot(loss)


# plt.tight_layout()

# plt.show()

