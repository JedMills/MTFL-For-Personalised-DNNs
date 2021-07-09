import numpy as np
import torch
from torch.optim import Optimizer
from models import NumpyModel



class ServerOpt():
    """
    Server optimizer base class for use with AdaptiveFedOpt.
    """
    
    def apply_gradients(self, model, grads):
        """
        Return copy of updated model.
        
        Args:
            - model: (NumpyModel) global model before step
            - grads: (NumpyModel) round psuedogradient
            
        Returns:
            (NumpyModel) updated model
        """
        raise NotImplementedError()



class ServerAdam(ServerOpt):
    """
    FedAdam server optimiser.
    """
    
    def __init__(self, params, lr, beta1, beta2, epsilon):
        """
        Returns a new ServerAdam instance. Uses params argument to initialise 
        1st and 2nd moment. Learning rate is fixed as per AdaptiveFedOpt paper, 
        not uses the learning rate schedule from original Adam paper.
        
        Args:
            - params:   (NumpyModel) copy of client model parameters
            - lr:       (float)      learning rate 
            - beta1:    (float)      1st moment estimate decay rate 
            - beta2:    (float)      2st moment estimate decay rate 
            - epsilon:  (float)      stability parameter  
        """
        self.m = params.zeros_like()
        self.v = params.zeros_like()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
    def apply_gradients(self, model, grads):
        """
        Return model with one step of Adam.
        
        Args:
            - model: (NumpyModel) global model before step
            - grads: (NumpyModel) round psuedogradient
            
        Returns:
            (NumpyModel) updated model
        """
        self.m = (self.beta1 * self.m) + (1 - self.beta1) * grads
        self.v = (self.beta2 * self.v) + (1 - self.beta2) * (grads ** 2)
        
        # uses constant learning rate as per AdaptiveFedOpt paper
        return model - (self.m * self.lr) / ((self.v ** 0.5) + self.epsilon)



class pFedMeOptimizer(Optimizer):
    """
    Optimizer to use for pFedMe simulations.
    """
    
    def __init__(self, params, device, lr=0.01, lamda=0.1 , mu = 0.001):
        """
        Return a new pFedMe optimizer. The passed mu parameter does not 
        explicitly feature in the pFedMe Algorithm, is used for weight decay.
        
        Args:
            - params: (iterable)        of nn.Module parameters
            - device: (torch.device)    where to place optimizer
            - lr:     (float)           learnign rate
            - lamda:  (float)           pFedMe lambda parameter
            - mu:     (float)           pFedMe mu parameter
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        
        defaults = dict(lr=lr, lamda=lamda, mu = mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)
        self.device = device
    
    def step(self, omega, closure=None):
        """
        One step of pFedMe. 
        
        Args:
            - omega: (NumpyModel) local model
        """
        loss = None
        if closure is not None:
            loss = closure
        
        # apply pFedMe update rule 
        for group in self.param_groups:
            for p, localweight in zip( group['params'], omega):
                w = torch.tensor(localweight).to(self.device)
                p.data = p.data - group['lr'] * (  p.grad.data 
                                                 + group['lamda'] * (p.data - w)
                                                 + group['mu'] * p.data)
        
        return  group['params'], loss



class ClientOpt():
    """
    Client optimiser base class for use with FedAvg/AdaptiveFedOpt.
    """
    
    def get_params(self):
        """
        Returns:
            (NumpyModel) copy of all optimiser parameters.
        """
        raise NotImplementedError()
    
    def set_params(self, params):
        """
        Set all optimiser parameters.
        
        Args:
            - params: (NumpyModel) values to set
        """
        raise NotImplementedError()

    def get_bn_params(self, setting=0):
        """
        Return only BN parameters. Setting can be one of the following 
        {0: usyb, 1: yb, 2: us, 3: none} to get different types of parameters.
        
        Args:
            - setting (int) param types to get
            
        Returns:
            list of numpy.ndarrays
        """
        raise NotImplementedError()
        
    def set_bn_params(self, params, setting=0):
        """
        Set only BN parameters. Setting can be one of the following 
        {0: usyb, 1: yb, 2: us, 3: none} to get different types of parameters.
        
        Args:
            - params  (list) of numpy.ndarray values to set
            - setting (int) param types to get
        """
        raise NotImplementedError()



class ClientSGD(torch.optim.SGD, ClientOpt):
    """
    Client SGD optimizer for FedAvg and AdaptiveFedOpt.
    """

    def __init__(self, params, lr):
        super(ClientSGD, self).__init__(params, lr)
        
    def get_params(self):
        """
        Returns:
            (NumpyModel) copy of all optimiser parameters.
        """
        return NumpyModel([])
        
    def set_params(self, params):
        """
        Set all optimiser parameters.
        
        Args:
            - params: (NumpyModel) values to set
        """
        pass
        
    def get_bn_params(self, model, setting=0):
        """
        Vanilla SGD has no optimisation parameters. Returns empty list.
        
        Returns:
            [] empty list.
        """
        return []
        
    def set_bn_params(self, params, model, setting=0):
        """
        Vanilla SGD has no optimisation parameters. Does nothing.
        """
        pass
        
    def step(self, closure=None, beta=None):
        """
        SGD step. 
        
        Args: 
            - beta: (float) optional different learning rate.
        """
        loss = None
        if closure is not None:
            loss = closure

        # apply SGD update rule
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if beta is None:
                    p.data.add_(d_p, alpha=-group['lr'])
                else:     
                    p.data.add_(d_p, alpha=-beta)
        
        return loss



class ClientAdam(torch.optim.Adam, ClientOpt):
    """
    Client Adam optimizer for FedAvg.
    """

    def __init__(   self, params, lr=0.001, betas=(0.9, 0.999), 
                    eps=1e-07, weight_decay=0, amsgrad=False):
        """
        Returns a new ClientAdam.
        
        Args:
            - params:      (NumpyModel) copy of client model parameters
            - lr:          (float)      learning rate 
            - betas:       (tuple)      two floats, 1st/2nd moment decay rates
            - eps:         (float)      stability parameter
            - weight_decay (float)      L2 decay rate
            - amsgrad      (bool)       whether to use amsgrad variant
        """
        super(ClientAdam, self).__init__(   params, lr, betas, eps, 
                                            weight_decay, amsgrad)
    
    def get_bn_params(self, model, setting=0):
        """
        Return only BN parameters. Setting can be one of the following 
        {0: usyb, 1: yb, 2: us, 3: none} to get different types of parameters.
        
        Args:
            - setting (int) param types to get
            
        Returns:
            list of numpy.ndarrays
        """
        if setting in [2, 3]:
            return []
        
        # order is (weight m, weight v, bias m, bias v)
        params = []
        for bn in model.bn_layers:
            weight = self.state[bn.weight]
            bias = self.state[bn.bias]
            params.append(np.copy(weight['exp_avg'].cpu().numpy()))
            params.append(np.copy(weight['exp_avg_sq'].cpu().numpy()))
            params.append(np.copy(bias['exp_avg'].cpu().numpy()))
            params.append(np.copy(bias['exp_avg_sq'].cpu().numpy()))
        
        return params
        
    def set_bn_params(self, params, model, setting=0):
        """
        Set only BN parameters. Setting can be one of the following 
        {0: usyb, 1: yb, 2: us, 3: none} to get different types of parameters.
        Order of parameters should be (weight m, weight v, bias m, bias v). 
        Length of params argument will then be 4*num_bn_layers.
        
        Args:
            - params  (list) of numpy.ndarray values to set
            - setting (int) param types to get
        """
        if setting in [2, 3]:
            return
        
        i = 0
        for bn in model.bn_layers:
            weight = self.state[bn.weight]
            bias = self.state[bn.bias]
            weight['exp_avg'].copy_(torch.tensor(params[i]))
            weight['exp_avg_sq'].copy_(torch.tensor(params[i+1]))
            bias['exp_avg'].copy_(torch.tensor(params[i+2]))
            bias['exp_avg_sq'].copy_(torch.tensor(params[i+3]))
            i += 4
        
    def get_params(self):
        """
        Order of values in returned NumpModel is (step_num, m, v), for each 
        model parameter.
        
        Returns:
            (NumpyModel) copy of all optimiser parameters.
        """
        params = []
        for key in self.state.keys():
            params.append(self.state[key]['step'])
            params.append(self.state[key]['exp_avg'].cpu().numpy())
            params.append(self.state[key]['exp_avg_sq'].cpu().numpy())
            
        return NumpyModel(params)
    
    def set_params(self, params):
        """
        Order of values in params arg should be (step_num, m, v), for each 
        model parameter.
        
        Args:
            (NumpyModel) parameters to set.
        """
        i = 0
        for key in self.state.keys():
            self.state[key]['step'] = params[i]
            self.state[key]['exp_avg'].copy_(torch.tensor(params[i+1]))
            self.state[key]['exp_avg_sq'].copy_(torch.tensor(params[i+2]))
            i += 3

            