from dataset import MakeData
import matplotlib.pyplot as plt

datamaker = MakeData()

x,y = datamaker.make_moons_data(200)

plt.scatter(x,y)
plt.show()