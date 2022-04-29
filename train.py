import numpy as np

def Train(self,optimizer_name,X,y,decay_factor = 0.9,  eps=0.0000001,b1 = 0.9, b2 = 0.999):
    X = self.add_ones(X.values)
    self.cost_history = np.zeros(self.epoch)
    self.theta = np.zeros(X.shape[1])
    self.cost_history = np.zeros(self.epoch)


    if optimizer_name == 'RMS':

        E = [0 for _ in range(len(self.theta))]
        for i in range(self.epoch):
        
            for idx, gradient in enumerate(X.T @ (X @ self.theta - y)):    
                E[idx] = decay_factor*E[idx] + (1 - decay_factor)*(gradient**2)
                self.theta[idx] = self.theta[idx] - (self.lr/np.sqrt(E[idx] + eps))*gradient

                self.cost_history[i] = self.compute_cost(X, y)

        

    elif optimizer_name == 'Adam':
    # Variable Initialization
        num_param = len(self.theta)
        m = [0 for _ in range(num_param)] # two m for each parameter
        v = [0 for _ in range(num_param)] # two v for each parameter
        g = [0 for _ in range(num_param)] # two gradient
        
        for t in range(1,self.epoch): 
            # Get the partial derivatives
            g = (X.T @ (X @ self.theta - y))

            # Update the m and v parameter
            m = [b1*m_i + (1 - b1)*g_i for m_i, g_i in zip(m, g)]
            v = [b2*v_i + (1 - b2)*(g_i**2) for v_i, g_i in zip(v, g)]

            # Bias correction for m and v
            m_cor = [m_i / (1 - (b1**t)) for m_i in m]
            v_cor = [v_i / (1 - (b2**t)) for v_i in v]

            # Update the parameter
            self.theta = [theta - (self.lr / (np.sqrt(v_cor_i) + eps))*m_cor_i for theta, v_cor_i, m_cor_i in zip(self.theta, v_cor, m_cor)]
            
            self.cost_history[t] = self.compute_cost(X, y)

    elif optimizer_name == 'Adagrad':
        num_param = len(self.theta)
        G = [[0 for _ in range(num_param)] for _ in range(num_param)]
        
        for i in range(self.epoch):
            
            # Update G and the model weights iteratively (Note: speed up could be gained from vectorized implementation)
            for idx, gradient in enumerate((X.T @ (X @ self.theta - y))):
                G[idx][idx] = G[idx][idx] + gradient**2
                self.theta[idx] = self.theta[idx] - (self.lr / np.sqrt(G[idx][idx] + eps)) * gradient
        
            self.cost_history[i] = self.compute_cost(X, y)

    elif optimizer_name == 'Adadelta':
            # Init Running Averages
        num_param = len(self.theta)
        E_g = [0 for _ in range(num_param)]
        E_p = [0 for _ in range(num_param)]
        delta_p = [0 for _ in range(num_param)]
        
        
        for i in range(self.epoch):
            
            # Select a random x and y
        # x, y = stochastic_sample(xs, ys)
        
            for idx, gradient in enumerate((X.T @ (X @ self.theta - y))):
                # Get the running average for the gradient
                E_g[idx] = decay_factor*E_g[idx] + (1 - decay_factor)*(gradient**2)
                
                # Get the running average for the parameters
                E_p[idx] = decay_factor*E_p[idx] + ((1 - decay_factor)*(delta_p[idx]**2))
                
                # Calculate the gradient difference
                delta_p[idx] = - np.sqrt(E_p[idx] + eps) / np.sqrt(E_g[idx] + eps) * gradient
                
                # update the model weight
                self.theta[idx] = self.theta[idx] + delta_p[idx]

                self.cost_history[i] = self.compute_cost(X, y)

    elif optimizer_name == 'Momentum':
        learning_rate = 2e-4
        # Create the gradient that we keep track as an array of 0 of the same size as the number of weights
        gradients = [0 for _ in range(len(self.theta))]
        
        for i in range(self.epoch):

            # Calculate the new gradients
            gradients = [(decay_factor * g) + (learning_rate * derivative) for g, derivative in zip(gradients, (X.T @ (X @ self.theta - y)))]
            
            # Updating the model parameters
            self.theta = [theta - g for theta, g in zip(self.theta, gradients)]

            self.cost_history[i] = self.compute_cost(X, y)

    return self.theta, self.cost_history