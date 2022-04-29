from importlib.metadata import metadata
from data import dataloader
from config import args
from model import LinearRegression
import optimizers
import utils



metadata = args.metadata
path = args.path

#Splitting data
X_train, X_test, Y_train, Y_test = dataloader.Dataset.split_data(metadata,0.8)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m','--optimizer_name', help = 'This is the name of the optimizer',required = True)
parser.add_argument('-n','--num_epochs', help = 'This is the number of epochs',type = int,required = True)

mains_args = vars(parser.parse_args())
epochs = mains_args['num_epochs']

model = LinearRegression(args.lr,epochs)
list_optimizer = ['Momentum','Adagrad','Adadelta','Adam','RMS']
list_lost = []

if mains_args['optimizer_name']=='RMS':
    _,loss_history = model.Train('RMS',X_train,Y_train)
    utils.Plot(loss_history,'RMS',epochs)
elif mains_args['optimizer_name']== 'Adam':
     _,loss_history = model.Train('Adam',X_train,Y_train)
     utils.Plot(loss_history,'Adam',epochs)
elif mains_args['optimizer_name']== 'Adagrad':
     _,loss_history = model.Train('Adagrad',X_train,Y_train)
     utils.Plot(loss_history,'Adagrad',epochs)  
elif mains_args['optimizer_name']== 'Adadelta':
     _,loss_history = model.Train('Adadelta',X_train,Y_train)
     utils.Plot(loss_history,'Adadelta',epochs)        
elif mains_args['optimizer_name']== 'Momentum':
     _,loss_history = model.Train('Momentum',X_train,Y_train)
     utils.Plot(loss_history,'Momentum',epochs)
elif mains_args['optimizer_name'] == 'all':
     for optim in list_optimizer:
        _,loss_history = model.Train(optim,X_train,Y_train)
        list_lost.append(loss_history) 
     utils.Plot_All(list_lost,epochs)  
               