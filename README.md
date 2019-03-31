# FATML-MiniProject, Spring 2019

# Experimenting with “Deep Learning for Case-Based Reasoning through Prototypes: A Neural Network that Explains Its Predictions”
Marius Arvinte, Mai Lee Chang, (Ethan) Yuqian Heng

Modifications of the code from https://github.com/OscarcarLi/PrototypeDL and the paper Li, Oscar, et al. "Deep learning for case-based reasoning through prototypes: A neural network that explains its predictions." Thirty-Second AAAI Conference on Artificial Intelligence. 2018 at https://arxiv.org/abs/1710.04806.

Tensorflow implementation. Python 3.x required.

Folder structure:
- "main" - Contains individual .py scripts named "CAE_XYZ.py" for each dataset. Can be directly run. NN parameters and number of prototypes are editable in the first few lines.

- "results" - Presaved, extended simulation results for each dataset. .xls files contain weight matrices of the last dense layer. The "img" folder in each case contains images of the autoencoder performance and the decoded prototype vectors.

Notes on datasets:
- MNIST data is ready for use in the "MNIST_data" folder
- EMNIST data is archived in the "EMNIST_data" folder. Unzip the .mat file directly in "EMNIST_data".
- CIFAR-10 data will be automaticatly downloaded in the "CIFAR_data" folder the first the "CAE_CIFAR10.py" script is run.
