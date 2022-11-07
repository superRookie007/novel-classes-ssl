'''Module to load cifar10 dataset.'''
import os
import os.path
import errno
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, check_integrity

from utils import data as data_utils
from architectures import VAE_MLP, VAE_CONV


class CIFAR10(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    validation_file = 'validation.pt'
    train_validation_file = 'train_validation.pt'

    def __init__(self, root, dataset, n_validation=5000, n_labeled=None, wanted_classes={0,1,2,3,4,5,6,7,8,9},
                 pure=True, weights=None, encoder=None, alpha=1.0, func='exp',
                 transform=None, target_transform=None, download=False, seed=0):
        if (dataset == 'test' or dataset == 'validation') and n_labeled:
            raise ValueError('n_labeled can only be used on training or train_validation set.')
        self.root = os.path.expanduser(root)
        self.dataset = dataset
        self.n_validation = n_validation
        self.n_labeled = n_labeled
        self.wanted_classes = wanted_classes

        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')


        if self.dataset == 'train':
            self.data, self.labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
            self.data, self.labels = data_utils.create_train(self.data, self.labels, self.n_labeled, wanted_labels=self.wanted_classes, pure=pure, seed=seed)
            self.labels[self.n_labeled:] = 0
            if weights=='raw':
                self.weights = data_utils.calculate_weights_knn(self.data/255, range(self.n_labeled), alpha=alpha, func=func, metric='euclidean', algorithm='auto', leaf_size=30)
            elif weights == 'encoding':
                if encoder == './vae_models/cifar10_vae_conv_10.pt':
                    model = VAE_CONV(latent_dim=10)
                elif encoder == './vae_models/cifar10_vae_conv_20.pt':
                    model = VAE_CONV(latent_dim=20)
                else:
                    model = VAE_MLP(latent_dim=10, dense_size=1000, img_dim=3*32*32)
                
                model.load_state_dict(torch.load(encoder))
                encodings = data_utils.get_encodings(model, self.data, name='cifar10', device='cuda', dim=3*32*32)
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
                if encoder == './vae_models/cifar10_vae_conv_10.pt':
                    model = VAE_CONV(latent_dim=10)
                elif encoder == './vae_models/cifar10_vae_conv_20.pt':
                    model = VAE_CONV(latent_dim=20)
                else:
                    model = VAE_MLP(latent_dim=10, dense_size=1000, img_dim=3*32*32)
                model.load_state_dict(torch.load(encoder))
                encodings = data_utils.get_encodings(model, self.data, name='cifar10', device='cuda', dim=3*32*32)
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


        self._load_meta() # load meta data


    def _load_pickled_arrays(self, train):
        if train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        
        data = []
        targets = []

        # now load the pickled numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                data.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))  # convert to HWC (PIL format)
        targets = np.array(targets)
        return torch.from_numpy(data), torch.from_numpy(targets)
        

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

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
        img = Image.fromarray(img.numpy(), mode='RGB')

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

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def assign_labels(self, pred):
        pred = pred.view(len(pred))
        self.labels[self.n_labeled:] = pred

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)
        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as err:
            if err.errno == errno.EEXIST:
                pass
            else:
                raise

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.root)
        
        # process and save as torch files
        print('Processing...')

        train_val_data, train_val_labels = self._load_pickled_arrays(train=True)
        # hard code the seed to make the split of training and validation sets constant.
        (train_data, train_label), (val_data, val_label) = data_utils.stratified_sampling(train_val_data, train_val_labels, self.n_validation, seed=0)

        test_data, test_labels = self._load_pickled_arrays(train=False)

        training_set = (train_data, train_label)
        validation_set = (val_data, val_label)
        train_validation_set = (train_val_data, train_val_labels)
        test_set = (test_data, test_labels)

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