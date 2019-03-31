import _pickle as cPickle
import numpy as np
import os
import tarfile
import zipfile
import sys

from urllib.request import urlretrieve
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# Import and serialize EMNIST data
def load_data_serialize(mat_file_path, val_ratio):
    mat = loadmat(mat_file_path)
    data = mat['dataset']
    mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
    enc = LabelEncoder()
    all_training_images = data['train'][0,0]['images'][0,0]
    all_training_labels = data['train'][0,0]['labels'][0,0]
    enc.fit(all_training_labels)
    all_training_labels_enc = enc.transform(all_training_labels)
    all_training_labels_onehot = np_utils.to_categorical(all_training_labels_enc)
    testing_images = data['test'][0,0]['images'][0,0]
    testing_labels = data['test'][0,0]['labels'][0,0]
    testing_labels_enc = enc.transform(testing_labels)
    testing_labels_onehot = np_utils.to_categorical(testing_labels_enc)
	
    # Convert type to float32
    all_training_images = all_training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize
    all_training_images /= 255
    testing_images /= 255
    
    # Split    
    training_images, val_images, training_labels, val_labels = train_test_split(all_training_images, all_training_labels_onehot, 
                                                        test_size=val_ratio, random_state=42)
    nb_classes = len(mapping)
    
    return ((training_images, training_labels), (val_images, val_labels), (testing_images, testing_labels_onehot), mapping, nb_classes, enc)

# Import and serialize CIFAR-10 data
# CIFAR-10 is automatically downloaded the first time
def get_data_set(name="train", style="gray"):
    x = None
    y = None

    maybe_download_and_extract()

    folder_name = "cifar_10"

    f = open('./CIFAR_data/'+folder_name+'/batches.meta', 'rb')
    f.close()

    if name == "train":
        for i in range(5):
            f = open('./CIFAR_data/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
            datadict = cPickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            if style == "gray":
                # Collapse one dimension by using RGB->gray conversion constants
                _Xx = 0.3*_X[:,:,:,0] + 0.59*_X[:,:,:,1] + 0.11*_X[:,:,:,2]
                _X  = _Xx
                _X  = _X.reshape(-1, 32*32)
            else:
                _X  = _X.reshape(-1, 32*32*3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

        # Split data in train / validation
        x_train, x_val = np.split(x, [int(0.8*x.shape[0])], axis=0)
        y_train, y_val = np.split(y, [int(0.8*y.shape[0])], axis=0)
        
        return x_train, dense_to_one_hot(y_train), x_val, dense_to_one_hot(y_val)

    elif name == "test":
        f = open('./CIFAR_data/'+folder_name+'/test_batch', 'rb')
        datadict = cPickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        if style == "gray":
            # Collapse one dimension by using RGB->gray conversion constants
            xx = 0.3*x[:,:,:,0] + 0.59*x[:,:,:,1] + 0.11*x[:,:,:,2]
            x  = xx
            x  = x.reshape(-1, 32*32)
        else:
            x = x.reshape(-1, 32*32*3)

    return x, dense_to_one_hot(y)

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

def maybe_download_and_extract():
    main_directory = "./CIFAR_data/"
    cifar_10_directory = main_directory+"cifar_10/"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.rename(main_directory+"./cifar-10-batches-py", cifar_10_directory)
        os.remove(zip_cifar_10)