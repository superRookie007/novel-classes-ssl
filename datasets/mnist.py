'''Module to load mnist dataset.'''
import os
import os.path
import errno
import codecs
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import data as data_utils
from architectures import VAE_MLP


class MNIST(Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        dataset (string): Can be one of 'train', 'validation', 'train_validation' and 'test'.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    validation_file = 'validation.pt'
    train_validation_file = 'train_validation.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
            '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, dataset, weights=None, encoder=None, n_validation=5000, 
        n_labeled=None, wanted_classes={0,1,2,3,4,5,6,7,8,9}, pure=True, 
        transform=None, target_transform=None, download=False, seed=0, 
        alpha=1.0, func='exp', dim=20):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset
        self.n_validation = n_validation
        self.n_labeled = n_labeled
        self.wanted_classes = wanted_classes

        if (dataset == 'test' or dataset == 'validation') and n_labeled:
            raise ValueError('n_labeled can only be used on training or train_validation set.')

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.dataset == 'train':
            self.data, self.labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
            self.data, self.labels = data_utils.create_train(self.data, self.labels, self.n_labeled, wanted_labels=self.wanted_classes, pure=pure, seed=seed)
            self.labels[self.n_labeled:] = 0
            if weights=='raw':
                self.weights = data_utils.calculate_weights_knn(self.data/255, range(self.n_labeled), alpha=alpha, func=func, metric='euclidean', algorithm='auto', leaf_size=30)
            elif weights == 'encoding':
                model = VAE_MLP(dim)
                model.load_state_dict(torch.load(encoder))
                encodings = data_utils.get_encodings(model, self.data, name=None, device='cuda')
                self.weights = data_utils.calculate_weights_knn(encodings, range(self.n_labeled), alpha=alpha, func=func, metric='euclidean', algorithm='auto', leaf_size=30)
            else:
                self.weights = torch.ones([len(self.data)], dtype=torch.float)
        elif self.dataset == 'train_validation':
            self.data, self.labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.train_validation_file))
            self.data, self.labels = data_utils.create_train(self.data, self.labels, self.n_labeled, wanted_labels=self.wanted_classes, pure=pure, seed=seed)
            self.labels[self.n_labeled:] = 0
            if weights=='raw':
                self.weights = data_utils.calculate_weights_knn(self.data/255, range(self.n_labeled), alpha=alpha, func=func, metric='euclidean', algorithm='auto', leaf_size=30)
            elif weights == 'encoding':
                model = VAE_MLP(dim)
                model.load_state_dict(torch.load(encoder))
                encodings = data_utils.get_encodings(model, self.data, name=None, device='cuda')
                self.weights = data_utils.calculate_weights_knn(encodings, range(self.n_labeled), alpha=alpha, func=func, metric='euclidean', algorithm='auto', leaf_size=30)
            else:
                self.weights = torch.ones([len(self.data)], dtype=torch.float)


        elif self.dataset == 'validation':
            self.data, self.labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.validation_file))
            self.data, self.labels = data_utils.get_wanted_classes(self.data, self.labels, self.wanted_classes)
        elif self.dataset == 'test':
            self.data, self.labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            self.data, self.labels = data_utils.get_wanted_classes(self.data, self.labels, self.wanted_classes)
        else:
            raise ValueError("dataset can only be one of 'train', 'validation', 'train_validation' and 'test'.")


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.dataset == 'validation' or self.dataset == 'test':
            img, target = self.data[index], self.labels[index]
        else:
            img, target, weight = self.data[index], self.labels[index], self.weights[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.dataset == 'validation' or self.dataset == 'test':
            return img, target
        else:
            return img, target, weight


    def __len__(self):
        return len(self.data)


    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.train_validation_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.validation_file))

    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def assign_labels(self, pred):
        pred = pred.view(len(pred))
        self.labels[self.n_labeled:] = pred


    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as err:
            if err.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        train_val_data = read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte'))
        train_val_labels = read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        # hard code the seed to make the split of training and validation sets constant.
        (train_data, train_label), (val_data, val_label) = data_utils.stratified_sampling(train_val_data, train_val_labels, self.n_validation, seed=0)

        training_set = (train_data, train_label)
        validation_set = (val_data, val_label)
        train_validation_set = (train_val_data, train_val_labels)
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.validation_file), 'wb') as f:
            torch.save(validation_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.train_validation_file), 'wb') as f:
            torch.save(train_validation_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.dataset
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)