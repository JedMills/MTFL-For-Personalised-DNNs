import numpy as np
import pickle
import torch
from progressbar import progressbar
from models import NumpyModel


def init_stats_arrays(T):
    """
    Returns:
        (tupe) of 4 numpy 0-filled float32 arrays of length T.
    """
    return tuple(np.zeros(T, dtype=np.float32) for i in range(4))



def run_fedavg_google(  data_feeders, test_data, model, server_opt, T, M, 
                        K, B, test_freq=1, bn_setting=0, noisy_idxs=[]):
    """
    Run the Adaptive Federated Optimization (AdaptiveFedOpt) algorithm from 
    'Adaptive Federated Optimization', Reddi et al., ICLR 2021. AdaptiveFedOpt 
    uses SGD on clients and a generic server optimizer to update the global 
    model each round. Runs T rounds of AdaptiveFedOpt, and returns the training 
    and test results.
    
    Args:
        - data_feeders: (list)      of PyTorchDataFeeders
        - test_data:    (tuple)     of (x, y) testing data as torch.tensors
        - model:        (FLModel)   to run SGD on 
        - server_opt    (ServerOpt) to update the global model on the server
        - T:            (int)       number of communication rounds
        - M:            (int)       number of clients sampled per round
        - K:            (int)       number of local training steps
        - B:            (int)       client batch size
        - test_freq:    (int)       how often to test UA
        - bn_setting:   (int)       private: 0=ybus, 1=yb, 2=us, 3=none
        - noisy_idxs:   (iterable)  indexes of noisy clients (ignore their UA)
        
    Returns:
        Tuple containing (train_errs, train_accs, test_errs, test_accs) as 
        Numpy arrays of length T. If test_freq > 1, non-tested rounds will 
        contain 0's.
    """
    W = len(data_feeders)
    
    train_errs, train_accs, test_errs, test_accs = init_stats_arrays(T)
    
    round_model = NumpyModel(model.get_params())
    round_grads = NumpyModel(model.get_params())
    
    # contains private BN vals (if bn_setting != 3)
    user_bn_model_vals = [model.get_bn_vals(bn_setting) for w in range(W)]
    
    for t in progressbar(range(T)):
        round_grads = round_grads.zeros_like()  # round psuedogradient
        
        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)
        weights   = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights   = weights.astype(np.float32)
        weights   /= np.sum(weights)
        
        round_n_test_users = 0
        
        for (w, user_idx) in zip(weights, user_idxs):
            # download global model, update local model with private BN params
            model.set_params(round_model)
            model.set_bn_vals(user_bn_model_vals[user_idx], bn_setting)
                        
            # test local model if not a noisy client
            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                err, acc = model.test(  test_data[0][user_idx], 
                                        test_data[1][user_idx], 128)
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1
            
            # perform local SGD
            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)
                loss, acc = model.train_step(x, y)
                train_errs[t] += loss
                train_accs[t] += acc

            # upload local model to server, store private BN params
            round_grads = round_grads + ((round_model - model.get_params()) * w)
            user_bn_model_vals[user_idx] = model.get_bn_vals(bn_setting)
        
        # update global model using psuedogradient
        round_model = server_opt.apply_gradients(round_model, round_grads)
        
        if t % test_freq == 0:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users
    
    train_errs /= M * K
    train_accs /= M * K
    
    return train_errs, train_accs, test_errs, test_accs



def run_fedavg( data_feeders, test_data, model, client_opt,  
                T, M, K, B, test_freq=1, bn_setting=0, noisy_idxs=[]):
    """
    Run Federated Averaging (FedAvg) algorithm from 'Communication-efficient
    learning of deep networks from decentralized data', McMahan et al., AISTATS 
    2021. In this implementation, the parameters of the client optimisers are 
    also averaged (gives FedAvg-Adam when client_opt is ClientAdam). Runs T 
    rounds of FedAvg, and returns the training and test results.
    
    Args:
        - data_feeders: (list)      of PyTorchDataFeeders
        - test_data:    (tuple)     of (x, y) testing data as torch.tensors
        - model:        (FLModel)   to run SGD on 
        - client_opt:   (ClientOpt) distributed client optimiser
        - T:            (int)       number of communication rounds
        - M:            (int)       number of clients sampled per round
        - K:            (int)       number of local training steps
        - B:            (int)       client batch size
        - test_freq:    (int)       how often to test UA
        - bn_setting:   (int)       private: 0=ybus, 1=yb, 2=us, 3=none
        - noisy_idxs:   (iterable)  indexes of noisy clients (ignore their UA)
        
    Returns:
        Tuple containing (train_errs, train_accs, test_errs, test_accs) as 
        Numpy arrays of length T. If test_freq > 1, non-tested rounds will 
        contain 0's.
    """
    W = len(data_feeders)
    
    train_errs, train_accs, test_errs, test_accs = init_stats_arrays(T)
    
    # contains private model and optimiser BN vals (if bn_setting != 3)
    user_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for w in range(W)]
    user_bn_optim_vals = [client_opt.get_bn_params(model) for w in range(W)]
    
    # global model/optimiser updated at the end of each round
    round_model = model.get_params()
    round_optim = client_opt.get_params()
    
    # stores accumulated client models/optimisers each round
    round_agg   = model.get_params()
    round_opt_agg = client_opt.get_params()
    
    for t in progressbar(range(T)):
        round_agg = round_agg.zeros_like()
        round_opt_agg = round_opt_agg.zeros_like()
        
        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)        
        weights   = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights   = weights.astype(np.float32)
        weights   /= np.sum(weights)
        
        round_n_test_users = 0
        
        for (w, user_idx) in zip(weights, user_idxs):
            # download global model/optim, update with private BN params
            model.set_params(round_model)
            client_opt.set_params(round_optim)
            model.set_bn_vals(user_bn_model_vals[user_idx], setting=bn_setting)
            client_opt.set_bn_params(user_bn_optim_vals[user_idx], 
                                        model, setting=bn_setting)
            
            # test local model if not a noisy client
            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                err, acc = model.test(  test_data[0][user_idx], 
                                        test_data[1][user_idx], 128)
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1
            
            # perform local SGD
            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)
                err, acc = model.train_step(x, y)
                train_errs[t] += err
                train_accs[t] += acc

            # upload local model/optim to server, store private BN params
            round_agg = round_agg + (model.get_params() * w)
            round_opt_agg = round_opt_agg + (client_opt.get_params() * w)
            user_bn_model_vals[user_idx] = model.get_bn_vals(setting=bn_setting)
            user_bn_optim_vals[user_idx] = client_opt.get_bn_params(model,
                                                setting=bn_setting)
            
        # new global model is weighted sum of client models
        round_model = round_agg.copy()
        round_optim = round_opt_agg.copy()
        
        if t % test_freq == 0:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users
    
    train_errs /= M * K
    train_accs /= M * K
    
    return train_errs, train_accs, test_errs, test_accs



def run_per_fedavg( data_feeders, test_data, model, beta, T, M, K, B, 
                    test_freq=1, noisy_idxs=[]):
    """
    Run Personalized-FedAvg (Per-FedAvg) algorithm from 'Personalized Federated
    Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning 
    Approach', Fallah  et al., NeurIPS 2020. Runs T rounds of Per-FedAvg, and 
    returns the test results. Note we are usign the first-order approximation 
    variant (i) described in Section 5 of Per-FedAvg paper.
    
    Args:
        - data_feeders: (list)      of PyTorchDataFeeders
        - test_data:    (tuple)     of (x, y) testing data as torch.tensors
        - model:        (FLModel)   to run SGD on 
        - beta:         (float)     parameter of Per-FedAvg algorithm
        - T:            (int)       number of communication rounds
        - M:            (int)       number of clients sampled per round
        - K:            (int)       number of local training steps
        - B:            (int)       client batch size
        - test_freq:    (int)       how often to test UA
        - noisy_idxs:   (iterable)  indexes of noisy clients (ignore their UA)
        
    Returns:
        Tuple containing (test_errs, test_accs) as Numpy arrays of length T. If 
        test_freq > 1, non-tested rounds will contain 0's.
    """
    W = len(data_feeders)
        
    train_errs, train_accs, test_errs, test_accs = init_stats_arrays(T)
    
    # global model updated at the end of each round, and round model accumulator 
    round_model = model.get_params()
    round_agg   = model.get_params()
    
    for t in progressbar(range(T), redirect_stdout=True):
        round_agg = round_agg.zeros_like()
        
        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)        
        weights   = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights   = weights.astype(np.float32)
        weights   /= np.sum(weights)
                
        round_n_test_users = 0
                
        for (w, user_idx) in zip(weights, user_idxs):
            # download global model
            model.set_params(round_model)
            
            # personalise global model and test, if not a noisy client
            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                x, y = data_feeders[user_idx].next_batch(B)
                model.train_step(x, y)
                err, acc = model.test(  test_data[0][user_idx], 
                                        test_data[1][user_idx], 
                                        128)
                test_errs[t]        += err
                test_accs[t]        += acc
                round_n_test_users  += 1
                model.set_params(round_model)
            
            # perform k steps of local training, as per Algorithm 1 of paper
            for k in range(K):
                start_model = model.get_params()
                
                x, y = data_feeders[user_idx].next_batch(B)
                loss, acc = model.train_step(x, y)
                
                logits = model.forward(x)
                loss = model.loss_fn(logits, y)
                model.optim.zero_grad()
                loss.backward()        
                model.optim.step()
                
                x, y = data_feeders[user_idx].next_batch(B)
                logits = model.forward(x)
                loss = model.loss_fn(logits, y)
                model.optim.zero_grad()
                loss.backward()
                
                model.set_params(start_model)
                model.optim.step(beta=beta)

            # add to round gradients
            round_agg = round_agg + (model.get_params() * w)
            
        # new global model is weighted sum of client models
        round_model = round_agg.copy()
                
        if t % test_freq == 0:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users
    
    return test_errs, test_accs



def run_pFedMe( data_feeders, test_data, model, T, M, K, B, R, lamda, eta, 
                beta, test_freq=1, noisy_idxs=[]):
    """
    Run pFedMe algorithm from 'Personalized Federated Learning with Moreau 
    Envelopes', Dinh et al., NeurIPS 2020. Runs T rounds of pFedMe, and returns 
    the test results. Note that, to make the algorithm comparison fair, we do 
    not activate all clients as per Algorithm 1 of the pFedMe paper, only 
    sending back the gradients of the sampled clients. Instead, we only activate 
    the sampled clients each round, brining pFedMe in line with the comparisons.
    
    Args:
        - data_feeders: (list)      of PyTorchDataFeeders
        - test_data:    (tuple)     of (x, y) testing data as torch.tensors
        - model:        (FLModel)   to run SGD on 
        - beta:         (float)     parameter of Per-FedAvg algorithm
        - T:            (int)       number of communication rounds
        - M:            (int)       number of clients sampled per round
        - K:            (int)       number of local training steps
        - B:            (int)       client batch size
        - R:            (int)       parameter R of pFedMe
        - lamda:        (float)     parameter lambda of pFedMe
        - eta:          (float)     learning rate of pFedMe
        - test_freq:    (int)       how often to test UA
        - noisy_idxs:   (iterable)  indexes of noisy clients (ignore their UA)
        
    Returns:
        Tuple containing (test_errs, test_accs) as Numpy arrays of length T. If 
        test_freq > 1, non-tested rounds will contain 0's.
    """
    W = len(data_feeders)
        
    train_errs, train_accs, test_errs, test_accs = init_stats_arrays(T)
    
    # global model updated at the end of each round, and round model accumulator 
    round_model = model.get_params()
    round_agg   = model.get_params()
    
    # client personalised models
    user_models = [round_model.copy() for w in range(W)]
    
    for t in progressbar(range(T), redirect_stdout=True):
        round_agg = round_agg.zeros_like()
        
        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)
        weights   = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights   = weights.astype(np.float32)
        weights   /= np.sum(weights)
        
        round_n_test_users  = 0
                
        for (w, user_idx) in zip(weights, user_idxs):

            # test local model if not a noisy client
            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                model.set_params(user_models[user_idx])
                err, acc = model.test(  test_data[0][user_idx], 
                                        test_data[1][user_idx], 
                                        128)
                test_errs[t]        += err
                test_accs[t]        += acc
                round_n_test_users  += 1

            # download global model
            model.set_params(round_model)
            
            # perform k steps of local training
            for r in range(R):
                x, y = data_feeders[user_idx].next_batch(B)
                omega = user_models[user_idx]
                for k in range(K):
                    model.optim.zero_grad()
                    logits = model.forward(x)
                    loss = model.loss_fn(logits, y)
                    loss.backward()        
                    model.optim.step(omega)
                    
                theta = model.get_params()
                
                user_models[user_idx] = omega - (lamda * eta * (omega - theta))
                
            round_agg = round_agg + (user_models[user_idx] * w)
            
        # new global model is weighted sum of client models
        round_model = (1 - beta) * round_model + beta * round_agg
        
        if t % test_freq == 0:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users
    
    return test_errs, test_accs
