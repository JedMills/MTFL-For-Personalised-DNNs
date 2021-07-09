# Multi-Task Federated Learning for Personalised Deep Neural Networks in Edge Computing 

This repository contains the code to run simulations from the [Multi-Task Federated Learning for Personalised Deep Neural Networks in Edge Computing](https://arxiv.org/abs/2007.09236), also published in IEEE TPDS journal.

Contains implementations of FedAvg, FedAvg-Adam, FedAdam [1], Per-FedAvg [2] and pFedMe [3] as described in the paper.

### Requirements
| Package      | Version |
| ------------ | ------- |
| python       | 3.7     |
| pytorch      | 1.7.0   |
| torchvision  | 0.8.1   |
| numpy        | 1.18.5  |
| progressbar2 | 3.47.0  |

### Data
Requires [MNIST](http:/yann.lecun.com/exdb/mnist/) and [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets. For MNIST, the .gz files should be saved in a folder with path `'../MNIST_data/'`, and for CIFAR10, the python pickle files should be saved in the folder `'../CIFAR10_data/'`.

### Running
Run main.py. Each experiment setting requires different command-line arguments. Will save a `.pkl` file in the same directory containing experiment data as numpy arrays. 


### References
[1] [_Adaptive Federated Optimization_](https://openreview.net/forum?id=LkFG3lB13U5), Reddi et al. ICLR 2021.

[2] [_Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach_](https://proceedings.neurips.cc/paper/2020/file/24389bfe4fe2eba8bf9aa9203a44cdad-Paper.pdf), Fallah et al. NeurIPS 2020. 

[3] [_Personalized Federated Learning with Moreau Envelopes_](https://proceedings.neurips.cc/paper/2020/file/f4f1f13c8289ac1b1ee0ff176b56fc60-Paper.pdf), Dinh et al. NeurIPS 2020.