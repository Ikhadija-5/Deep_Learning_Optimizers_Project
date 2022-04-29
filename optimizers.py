from ast import arg
import numpy as np
import model
from config import args


#RMSPROP
lr = args.lr
epoch = args.epoch
def RMS(X,y,decay_factor = 0.9,  eps=0.0000001):  
    theta = np.zeros(X.shape[1])
    cost_history = np.zeros(epoch)

    E = [0 for _ in range(len(theta))]
    for i in range(epoch):
        for idx, gradient in enumerate(X.T @ (X @ theta - y)):    
            E[idx] = decay_factor*E[idx] + (1 - decay_factor)*(gradient**2)
            theta[idx] = theta[idx] - (lr/np.sqrt(E[idx] + eps))*gradient

        cost_history[i] = model.LinearRegression.compute_cost(X, y)

    return theta, cost_history

#ADAM OPTIMIZER
def Adam(X,y,b1 = 0.9, b2 = 0.999,  eps=0.0000001):
    X = model.LinearRegression.add_ones(X.values)
    theta = np.zeros(X.shape[1])
    cost_history = np.zeros(epoch)

    # Variable Initialization
    num_param = len(theta)
    m = [0 for _ in range(num_param)] # two m for each parameter
    v = [0 for _ in range(num_param)] # two v for each parameter
    g = [0 for _ in range(num_param)] # two gradient

    for t in range(1,epoch): 
        # Get the partial derivatives
        g = (X.T @ (X @ theta - y))

        # Update the m and v parameter
        m = [b1*m_i + (1 - b1)*g_i for m_i, g_i in zip(m, g)]
        v = [b2*v_i + (1 - b2)*(g_i**2) for v_i, g_i in zip(v, g)]

        # Bias correction for m and v
        m_cor = [m_i / (1 - (b1**t)) for m_i in m]
        v_cor = [v_i / (1 - (b2**t)) for v_i in v]

        # Update the parameter
        theta = [theta - (lr / (np.sqrt(v_cor_i) + eps))*m_cor_i for theta, v_cor_i, m_cor_i in zip(theta, v_cor, m_cor)]
        
        cost_history[t] = model.LinearRegression.compute_cost(X, y)

    return theta, cost_history



#ADAGRAD OPTIMIZER
def Adagrad(X, y, eps=0.0000001):
    m = len(y)
    X = model.LinearRegression.add_ones(X.values)
    theta = np.zeros(X.shape[1])
    cost_history = np.zeros(epoch)

    # Here only the diagonal matter
    num_param = len(theta)
    G = [[0 for _ in range(num_param)] for _ in range(num_param)]

    for i in range(epoch):
        
        # Select a random x and y
        # x, y = stochastic_sample(xs, ys)
        
        # Update G and the model weights iteratively (Note: speed up could be gained from vectorized implementation)
        for idx, gradient in enumerate((X.T @ (X @ theta - y))):
            G[idx][idx] = G[idx][idx] + gradient**2
            theta[idx] = theta[idx] - (lr / np.sqrt(G[idx][idx] + eps)) * gradient
        
        cost_history[i] = model.LinearRegression.compute_cost(X, y)

    return theta, cost_history

#ADADELTA OPTIMIZER
def Adadelta(X, y, decay_factor = 0.9, eps=0.0000001):
    X = model.LinearRegression.add_ones(X.values)
    theta = np.zeros(X.shape[1])
    cost_history = np.zeros(epoch)        
      
    # Init Running Averages
    num_param = len(theta)
    E_g = [0 for _ in range(num_param)]
    E_p = [0 for _ in range(num_param)]
    delta_p = [0 for _ in range(num_param)]
    
    
    for i in range(epoch):
        
        # Select a random x and y
      # x, y = stochastic_sample(xs, ys)
      
      for idx, gradient in enumerate((X.T @ (X @ theta - y))):
        # Get the running average for the gradient
        E_g[idx] = decay_factor*E_g[idx] + (1 - decay_factor)*(gradient**2)
        
        # Get the running average for the parameters
        E_p[idx] = decay_factor*E_p[idx] + ((1 - decay_factor)*(delta_p[idx]**2))
        
        # Calculate the gradient difference
        delta_p[idx] = - np.sqrt(E_p[idx] + eps) / np.sqrt(E_g[idx] + eps) * gradient
        
        # update the model weight
        theta[idx] = theta[idx] + delta_p[idx]

        cost_history[i] =model.LinearRegression.compute_cost(X, y)

    return theta, cost_history

#MOMENTUM
def Momentun( X, y, decay_factor = 0.9):
    X = model.LinearRegression.add_ones(X.values)
    theta = np.zeros(X.shape[1])
    cost_history = np.zeros(epoch)       
    # self.decay_factor = decay_factor 

    # Create the gradient that we keep track as an array of 0 of the same size as the number of weights
    gradients = [0 for _ in range(len(theta))]
    
    for i in range(epoch):

        # Calculate the new gradients
        gradients = [(decay_factor * g) + (lr * derivative) for g, derivative in zip(gradients, (X.T @ (X @ theta - y)))]
        
        # Updating the model parameters
        theta = [theta - g for theta, g in zip(theta, gradients)]

        cost_history[i] = model.LinearRegression.compute_cost(X, y)

    return theta, cost_history