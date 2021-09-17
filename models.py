import torch 
import numpy as np
import operator
import numbers


class FLModel(torch.nn.Module):
    """
    Has methods to easily allow setting of model parameters, performing training 
    steps, and retrieving model parameters.
    """
        
    def __init__(self, device):
        """
        Args:
            - device: (torch.device) to place model on
        """
        super(FLModel, self).__init__()
        self.optim      = None
        self.device     = device
        self.loss_fn    = None
        self.bn_layers  = []        # any model BN layers must be added to this 

    def set_optim(self, optim, init_optim=True):
        """
        Set the optimizer that this model will perform SGD with.
        
        Args:
            - optim:     (torch.optim) that model will perform SGD steps with
            - init_optim (bool)        whether to initialise optimiser params
        """
        self.optim = optim
        if init_optim:
            self.empty_step()
        
    def empty_step(self):
        """
        Perform one step of SGD with all-0 inputs and targets to initialise 
        optimiser parameters.
        """
        raise NotImplementedError()

    def get_params(self):
        """
        Returns model values as NumpyModel. BN layer statistics (mu, sigma) are
        added as parameters at the end of the returned model. 
        """
        ps = [np.copy(p.data.cpu().numpy()) for p in list(self.parameters())]
        for bn in self.bn_layers:
            ps.append(np.copy(bn.running_mean.cpu().numpy()))
            ps.append(np.copy(bn.running_var.cpu().numpy()))
        
        return NumpyModel(ps)
    
    def get_bn_vals(self, setting=0):
        """
        Returns the parameters from BN layers. Setting can be one of the 
        following {0: usyb, 1: yb, 2: us, 3: none} to get different types of 
        parameters.
        
        Args:
            - setting: (int) BN values to return 
            
        Returns:
            list of [np.ndarrays] containing BN parameters
        """
        if setting not in [0, 1, 2, 3]:
            raise ValueError('Setting must be in: {0, 1, 2, 3}')
    
        vals = []
        
        if setting == 3:
            return vals
        
        with torch.no_grad():
            # add gamma, beta
            if setting in [0, 1]:
                for bn in self.bn_layers:
                    vals.append(np.copy(bn.weight.cpu().numpy()))
                    vals.append(np.copy(bn.bias.cpu().numpy()))
            
            # add mu, sigma
            if setting in [0, 2]:
                for bn in self.bn_layers:
                    vals.append(np.copy(bn.running_mean.cpu().numpy()))
                    vals.append(np.copy(bn.running_var.cpu().numpy()))
        return vals


    def set_bn_vals(self, vals, setting=0):
        """
        Set the BN parameterss of the model. Setting can be one of the following
        {0: usyb, 1: yb, 2: us, 3: none}.
        
        Args:
            - vals:     (NumpyModel) new BN values to set
            - setting:  (int)        type of values to return
        """
        if setting not in [0, 1, 2, 3]:
            raise ValueError('Setting must be in: {0, 1, 2, 3}')
        
        if setting == 3:
            return
        
        with torch.no_grad():
            i = 0
            # set gamma, beta
            if setting in [0, 1]:
                for bn in self.bn_layers:
                    bn.weight.copy_(torch.tensor(vals[i]))
                    bn.bias.copy_(torch.tensor(vals[i+1]))
                    i += 2
                    
            # set mu, sigma
            if setting in [0, 2]:
                for bn in self.bn_layers:
                    bn.running_mean.copy_(torch.tensor(vals[i]))
                    bn.running_var.copy_(torch.tensor(vals[i+1]))
                    i += 2
    
    def set_params(self, params):
        """
        Passed params should be in the order of the model layers, as returned by
        get_params(), with the BN layer statistics (mu, sigma) appended to the 
        end of the model.
        
        Args:
            - params:   (NumpyModel) to set model values with
        """
        i = 0
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.tensor(params[i]))
                i += 1
                
            # set mu, sigma
            for bn in self.bn_layers:
                bn.running_mean.copy_(torch.tensor(params[i]))
                bn.running_var.copy_(torch.tensor(params[i+1]))
                i += 2
   
    def forward(self, x):
        """
        Returns outputs of model given data x.
        
        Args:
            - x: (torch.tensor) must be on same device as model
            
        Returns:
            torch.tensor model outputs
        """
        raise NotImplementedError()
        
    def calc_acc(self, logits, y):
        """
        Calculate accuracy/performance metric of model.
        
        Args:
            - logits: (torch.tensor) unnormalised predictions of y
            - y:      (torch.tensor) true values
            
        Returns:
            torch.tensor containing scalar value.
        """
        raise NotImplementedError()
    
    def train_step(self, x, y):
        """
        Perform one step of SGD using assigned optimizer.
        
        Args:
            - x: (torch.tensor) inputs
            - y: (torch.tensor) targets
        
        Returns:
            tupe of floats (loss, acc) calculated during the training step.
        """
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = self.calc_acc(logits, y)
        self.optim.zero_grad()
        loss.backward()        
        self.optim.step()
        
        return loss.item(), acc

    def test(self, x, y, B):
        """
        Calculate error and accuracy of passed data using batches of size B.
        
        Args:
            - x: (torch.tensor) inputs
            - y: (torch.tensor) labels
            - B: (int)          batch size
        
        Returns: 
            tuple of floats (loss, acc) averaged over passed data.
        """
        self.eval()
        n_batches = int(np.ceil(x.shape[0] / B))
        loss = 0.0
        acc = 0.0
        
        with torch.no_grad():
            for b in range(n_batches):
                logits = self.forward(x[b*B:(b+1)*B])
                loss += self.loss_fn(logits, y[b*B:(b+1)*B]).item()
                acc += self.calc_acc(logits, y[b*B:(b+1)*B])
        self.train()
        
        return loss/n_batches, acc/n_batches



class MNISTModel(FLModel):
    """
    2-hidden-layer fully connected model, 2 hidden layers with 200 units and a 
    BN layer. Categorical Cross Entropy loss.
    """
    
    def __init__(self, device):
        """
        Returns a new MNISTModelBN.
        
        Args:
            - device: (torch.device) to place model on
        """
        super(MNISTModel, self).__init__(device)
        
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        
        self.fc0    = torch.nn.Linear(784, 200).to(device)
        self.relu0  = torch.nn.ReLU().to(device)
        
        self.fc1    = torch.nn.Linear(200, 200).to(device)
        self.relu1  = torch.nn.ReLU().to(device)
        
        self.out    = torch.nn.Linear(200, 10).to(device)

        self.bn0 = torch.nn.BatchNorm1d(200).to(device)        
        
        self.bn_layers = [self.bn0]
        
    def forward(self, x):
        """
        Returns outputs of model given data x.
        
        Args:
            - x: (torch.tensor) must be on same device as model
            
        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """
        a = self.bn0(self.relu0(self.fc0(x)))
        b = self.relu1(self.fc1(a))
        
        return self.out(b)
        
    def calc_acc(self, logits, y):
        """
        Calculate top-1 accuracy of model.
        
        Args:
            - logits: (torch.tensor) unnormalised predictions of y
            - y:      (torch.tensor) true values
            
        Returns:
            torch.tensor containing scalar value.
        """
        return (torch.argmax(logits, dim=1) == y).float().mean()
        
    def empty_step(self):
        """
        Perform one step of SGD with all-0 inputs and targets to initialse 
        optimiser parameters.
        """
        self.train_step(torch.zeros((2, 784), 
                                    device=self.device, 
                                    dtype=torch.float32), 
                        torch.zeros((2), 
                                    device=self.device,
                                    dtype=torch.int32).long())



class CIFAR10Model(FLModel):
    """
    Convolutional model with two (Conv -> ReLU -> MaxPool -> BN) blocks, and one 
    fully connected hidden layer. Categorical Cross Entropy loss.
    """
    
    def __init__(self, device):
        """
        Returns a new CIFAR10Model.
        
        Args:
            - device: (torch.device) to place model on
        """
        super(CIFAR10Model, self).__init__(device)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        
        self.conv0 = torch.nn.Conv2d(3, 32, 3, 1).to(device)
        self.relu0 = torch.nn.ReLU().to(device)
        self.pool0 = torch.nn.MaxPool2d(2, 2).to(device)
        
        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1).to(device)
        self.relu1 = torch.nn.ReLU().to(device)
        self.pool1 = torch.nn.MaxPool2d(2, 2).to(device)
        
        self.flat  = torch.nn.Flatten().to(device)
        self.fc0   = torch.nn.Linear(2304, 512).to(device)
        self.relu2 = torch.nn.ReLU().to(device)
        
        self.out   = torch.nn.Linear(512, 10).to(device)
        
        self.bn0   = torch.nn.BatchNorm2d(32).to(device)
        self.bn1   = torch.nn.BatchNorm2d(64).to(device)
        
        self.bn_layers = [self.bn0, self.bn1]
        
    def forward(self, x):
        """
        Returns outputs of model given data x.
        
        Args:
            - x: (torch.tensor) must be on same device as model
            
        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """
        a = self.bn0(self.pool0(self.relu0(self.conv0(x))))
        b = self.bn1(self.pool1(self.relu1(self.conv1(a))))
        c = self.relu2(self.fc0(self.flat(b)))
        
        return self.out(c)
        
    def calc_acc(self, logits, y):
        """
        Calculate top-1 accuracy of model.
        
        Args:
            - logits: (torch.tensor) unnormalised predictions of y
            - y:      (torch.tensor) true values
            
        Returns:
            torch.tensor containing scalar value.
        """
        return (torch.argmax(logits, dim=1) == y).float().mean()
        
    def empty_step(self):
        """
        Perform one step of SGD with all-0 inputs and targets to initialise 
        optimiser parameters.
        """
        self.train_step(torch.zeros((2, 3, 32, 32), 
                                    device=self.device, 
                                    dtype=torch.float32), 
                        torch.zeros((2), 
                                    device=self.device,
                                    dtype=torch.int32).long())



class NumpyModel():
    """
    Allows easy operations on whole model of parameters using numpy arrays.
    """
    
    def __init__(self, params):
        """
        Return a NumpyModel using given params (values are not copied).
        
        Args:
            - params: (list) of numpy arrays representing model parameters.
        """
        self.params = params
        
    def copy(self):
        """
        Returns: 
            (NumpyModel) with all parameters copied from this model.
        """
        return NumpyModel([np.copy(p) for p in self.params])
        
    def zeros_like(self):
        """
        Returns:
            (NumpyModel) with all-0 values.
        """
        return NumpyModel([np.zeros_like(p) for p in self.params])
        
    def _op(self, other, f):
        """
        Return a new NumpyModel, where each parameter is computed using function
        f of this model's parameters and the other model's corresponding
        parameters/a constant value.
        
        Args:
            - other: (NumpyModel) or float/int 
            - f:     (function)   to apply
        """
        if np.isscalar(other):
            new_params = [f(p, other) for p in self.params]
            
        elif isinstance(other, NumpyModel):
            new_params = [f(p, o) for (p, o) in zip(self.params, other.params)]
            
        else:
            raise ValueError('Incompatible type for op: {}'.format(other))
        
        return NumpyModel(new_params)
        
        
    def abs(self):
        """
        Returns:
            (NumpyModel) with all absolute values.
        """
        return NumpyModel([np.absolute(p) for p in self.params])
        
    def __add__(self, other):
        """
        Args:
            - other: (NumpyModel) or float/int.
            
        Returns:
            (NumpyModel) of self + other, elementwise.
        """
        return self._op(other, operator.add)
        
    def __radd__(self, other):
        """
        Returns new NumpyModel with vals of (self + other).
        
        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.add)

    def __sub__(self, other):
        """
        Returns new NumpyModel with vals of (self - other).
        
        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.sub)
        
    def __mul__(self, other):
        """
        Returns new NumpyModel with vals of (self * other).
        
        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.mul)
        
    def __rmul__(self, other):
        """
        Returns new NumpyModel with vals of (self * other).
        
        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.mul)
        
    def __truediv__(self, other):
        """
        Returns new NumpyModel with vals of (self / other).
        
        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.truediv)
        
    def __pow__(self, other):
        """
        Returns new NumpyModel with vals of (self ^ other).
        
        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.pow)
        
    def __getitem__(self, key):
        """
        Get parameter [key] of model.
        """
        return self.params[key]
        
    def __len__(self):
        """
        Return number of parameters in model.
        """
        return len(self.params)
        
    def __iter__(self):
        """
        Iterate over parameters of model.
        """
        for p in self.params:
            yield p
