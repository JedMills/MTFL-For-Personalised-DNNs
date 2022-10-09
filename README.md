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
main.py runs a simulation for a single combination of dataset, algorithm and hyperparameters. Each experiment setting requires different command-line arguments. Will save a `.pkl` file in the same directory containing experiment data as numpy arrays. Example settings to reproduce some of the experiments from the paper are as follows:

FL(FedAvg) on MNIST with 200 workers, C=0.5 participation fraction, for 500 rounds on the GPU (top-left result of Table 2):
`python main.py -dset mnist -alg fedavg -C 0.5 -B 20 -T 500 -E 1 -device gpu -W 200 -seed 0 -lr 0.1 -noisy_frac 0.0 -bn_private none`

MTFL(FedAdam) with private u,s,y,B on MNIST with 400 workers, C=1.0 participation rate (also from Table 1), using client learning rate of 0.3, server learning rate 0.01, server Adam parameters beta1=0.9, beta2=0.999, epsilon=1e-4:
`python main.py -dset mnist -alg fedadam -C 1.0 -B 20 -T 500 -E 1 -device gpu -W 400 -seed 0 -lr 0.3 -noisy_frac 0.0 -bn_private usyb -beta1 0.9 -beta2 0.999 -epsilon 1e-4 -server_lr 0.01`

MTFL(FedAvg-Adam) with private y,b on CIFAR10, 400 workers, C=0.5 participation rate, (also from Table 1), with client learning rate 0.003, client Adam parameters beta1=0.9, beta2=0.999, epsilon=1e-7:
`python main.py -dset cifar10 -alg fedavg-adam -C 0.5 -B 20 -T 500 -E 1 -device gpu -W 400 -seed 0 -lr 0.003 -noisy_frac 0.0 -bn_private usyb -beta1 0.9 -beta2 0.999 -epsilon 1e-7`

pFedMe on MNIST with 200 workers and client participation rate 0.5 (shown in Fig. 5 a) for 200 rounds:
`python main.py -dset mnist -alg pfedme -C 0.5 -B 20 -T 200 -E 1 -device gpu -W 200 -seed 0 -lr 0.3 -noisy_frac 0.0 -beta 1.0 -lamda 1.0`

Per-FedAvg on CIFAR10 with 400 workers, client participation rate 1.0, for 200 rounds (Fig. 5 h):
`python main.py -dset cifar10 -alg perfedavg -C 1.0 -B 20 -T 200 -E 1 -device gpu -W 200 -seed 0 -lr 0.1 -noisy_frac 0.0 -beta 0.1`


### Output
Data is saved in '.pkl' files. Each file contains a dictionary, with the keys of the dictionary being the random seeds run for the given settings. If a file already exists in the run directory, the data for the run seed will be added to the existing file. If that seed already exists in the file, the dictionary entry with that key (seed) will be overwritten. For the FedAvg, FedAdam, FedAvg-Adam algorithms, each dictionary entry is a tuple of 4 values:
`(training_errors, training_accuracies, test_errors, test_accuracies)`
For pFedMe and Per-FedAvg, each dictionary entry is a tuple of 2 values:
`(test_errors, test_accuracies)`
Each item in the tuples is a numpy array of length T. The training error and accuracy arrays report the average err/acc over the K local steps for the selected clients.

### Hyperparameters 
The `Fig-5-settings.md` file contains a table listing all the hyperparameters used in the experiments shown in Fig. 5 of the MTFL paper. These settings are the same as the settings used in Table 2 and Table 3. For example,  the file lists the settings for {MNIST, W=200, C=0.5, MTFL(FedAvg), private=yb} as lr=0.3. The learning rates for {MNIST, W=200, C=0.5, MTFL(FedAvg), private=us} and {MNIST, W=200, C=0.5, MTFL(FedAvg), private=usyb} were also lr=0.3.


### References
[1] [_Adaptive Federated Optimization_](https://openreview.net/forum?id=LkFG3lB13U5), Reddi et al. ICLR 2021.

[2] [_Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach_](https://proceedings.neurips.cc/paper/2020/file/24389bfe4fe2eba8bf9aa9203a44cdad-Paper.pdf), Fallah et al. NeurIPS 2020. 

[3] [_Personalized Federated Learning with Moreau Envelopes_](https://proceedings.neurips.cc/paper/2020/file/f4f1f13c8289ac1b1ee0ff176b56fc60-Paper.pdf), Dinh et al. NeurIPS 2020.
