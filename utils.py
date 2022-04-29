from ast import arg
import matplotlib.pyplot as plt
from config import args
import numpy as np


def Plot_All(list_of_loss, epochs):
    plt.plot(np.arange(epochs), list_of_loss[0], label = "Momentum")
    plt.plot(np.arange(epochs), list_of_loss[1], label = "Adagrad")
    plt.plot(np.arange(epochs), list_of_loss[2], label = "Adadelta")    #['Momentum','Adagrad','Adadelta','Adam','RMS']
    plt.plot(np.arange(epochs), list_of_loss[3], label = "Adam")
    plt.plot(np.arange(epochs), list_of_loss[4], label = "RMS")
    plt.xlabel('Iterations')
    plt.ylabel('Cost, ' + r'$J(\theta)$')
    plt.legend()
    plt.savefig("./figures/All_plots.png")
    plt.show()

def Plot(loss_history,optim_name,epoch):
    plt.figure()
    plt.plot(np.arange(epoch), loss_history, c='green')
    plt.xlabel('Iterations')
    plt.ylabel('Cost, ' + r'$J(\theta)$')
    plt.savefig(f"./figures/{optim_name}.png")
    plt.show()    