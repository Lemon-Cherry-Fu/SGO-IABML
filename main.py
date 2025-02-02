
import torch

from torchvision.datasets import ImageFolder
from torchvision import transforms

import numpy as np
import os
import argparse

# import regression data generator
from RegressionDataset import SineDataset, LineDataset

from _utils import train_val_split, train_val_split_regression

from SGO_IABML import SGO_IABML

from EpisodeSampler import EpisodeSampler

# --------------------------------------------------
# SETUP INPUT PARSER
# --------------------------------------------------
parser = argparse.ArgumentParser(description='Setup variables')

parser.add_argument('--ds-folder', type=str, default='./data', help='Parent folder containing the dataset')
parser.add_argument('--datasource', action='append',
                    help='List of datasets: SineLine for regression, and Pumi, MetaDataset, MiniImageNet, TieredImageNet, and CIFAR for classification')

parser.add_argument('--img-size', action='append', help='A pair of image size: 32 or 84')

parser.add_argument('--ml-algorithm', type=str, default='SGO_IABML',
                    help='Few-shot learning methods, including: VAMPIRE, ABPML, VERSA, SImPa, MetaKernel, SGO_IABML')

parser.add_argument('--first-order', dest='first_order', action='store_true')
parser.add_argument('--no-first-order', dest='first_order', action='store_false')
parser.set_defaults(first_order=True)
parser.add_argument('--KL-weight', type=float, default=1e-6,
                    help='Weighting factor for the KL divergence')

parser.add_argument('--network-architecture', type=str, default='ResNet12',
                    help='The base model used, including CNN, ResNet12, and ViT-SUN defined in CommonModels')

# Including learnable BatchNorm in the model or not learnable BN
parser.add_argument('--batchnorm', dest='batchnorm', action='store_true')
parser.add_argument('--no-batchnorm', dest='batchnorm', action='store_false')
parser.set_defaults(batchnorm=False)

# use strided convolution or max-pooling
parser.add_argument('--strided', dest='strided', action='store_true')
parser.add_argument('--no-strided', dest='strided', action='store_false')
parser.set_defaults(strided=True)

parser.add_argument("--dropout-prob", type=float, default=0, help="Dropout probability")

parser.add_argument('--num-ways', type=int, default=5, help='Number of classes within a task')

parser.add_argument('--num-inner-updates', type=int, default=5,
                    help='The number of gradient updates for episode adaptation')
parser.add_argument('--inner-lr', type=float, default=0.01, help='Learning rate of episode adaptation step')

parser.add_argument('--logdir', type=str, default='./log', help='Folder to store model and logs')

parser.add_argument('--meta-lr', type=float, default=1e-3, help='Learning rate for meta-update')
parser.add_argument('--minibatch', type=int, default=20, help='Minibatch of episodes to update meta-parameters')

parser.add_argument('--k-shot', type=int, default=1, help='Number of training examples per class')
parser.add_argument('--v-shot', type=int, default=15, help='Number of validation examples per class')

parser.add_argument('--num-episodes-per-epoch', type=int, default=10000,
                    help='Save meta-parameters after this number of episodes')
parser.add_argument('--num-epochs', type=int, default=1, help='')
parser.add_argument('--resume-epoch', type=int, default=0, help='Resume')

parser.add_argument('--train', dest='train_flag', action='store_true')
parser.add_argument('--test', dest='train_flag', action='store_false')
parser.set_defaults(train_flag=True)

parser.add_argument('--num-workers', type=int, default=2, help='Number of workers used in data loader')

parser.add_argument('--num-models', type=int, default=1, help='Number of base network sampled from the hyper-net')

parser.add_argument('--num-episodes', type=int, default=100, help='Number of episodes used in testing')

args = parser.parse_args()
print()

config = {}
for key in args.__dict__:
    config[key] = args.__dict__[key]

config['logdir'] = os.path.join(config['logdir'], 'meta_learning', config['ml_algorithm'].lower(),
                                config['network_architecture'], config['datasource'][0])
if not os.path.exists(path=config['logdir']):
    from pathlib import Path

    Path(config['logdir']).mkdir(parents=True, exist_ok=True)

# config['minibatch_print'] = np.lcm(config['minibatch'], 1000)
config['minibatch_print'] = 2


config['device'] = torch.device('cuda:0') if torch.cuda.is_available() \
    else torch.device('cpu')

if __name__ == "__main__":
    if 'SineLine' not in config['datasource']:
        # define some transformation
        transformations = transforms.Compose(
            transforms=[
                transforms.Resize(size=([int(i) for i in config['img_size']])),
                transforms.ToTensor()
            ]
        )
        # classification
        if config['train_flag']:
            # training dataset
            train_dataset = torch.utils.data.ConcatDataset(
                datasets=[ImageFolder(
                    root=os.path.join(config['ds_folder'], data_source, 'train'),
                    transform=transformations
                ) for data_source in config['datasource']]
            )

            # training data loader
            train_dataloader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_sampler=EpisodeSampler(
                    sampler=torch.utils.data.RandomSampler(data_source=train_dataset),
                    num_ways=config['num_ways'],
                    drop_last=True,
                    num_samples_per_class=config['k_shot'] + config['v_shot']
                ),
                num_workers=config['num_workers'],
                pin_memory=True
            )

        # testing/validation data loader
        test_dataset = torch.utils.data.ConcatDataset(
            datasets=[ImageFolder(
                root=os.path.join(config['ds_folder'], data_source, 'val' if config['train_flag'] else 'test'),
                transform=transformations
            ) for data_source in config['datasource']]
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_sampler=EpisodeSampler(
                sampler=torch.utils.data.RandomSampler(data_source=test_dataset),
                num_ways=config['num_ways'],
                drop_last=True,
                num_samples_per_class=config['k_shot'] + config['v_shot']
            ),
            num_workers=2,
            pin_memory=True
        )

        config['loss_function'] = torch.nn.CrossEntropyLoss()
        config['train_val_split_function'] = train_val_split
    else:  # regression
        regression_dataset = torch.utils.data.ConcatDataset(
            datasets=[
                SineDataset(amplitude_range=[0.1, 5], phase_range=[0, np.pi], noise_std=0.3, x_range=[-5, 5],
                            num_samples=50),
                LineDataset(slope_range=[-3, 3], intercept_range=[-3, 3], x_range=[-5, 5], num_samples=50,
                            noise_std=0.3)
            ]
        )

        train_dataloader = torch.utils.data.DataLoader(dataset=regression_dataset, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(dataset=regression_dataset, shuffle=True)

        config['loss_function'] = torch.nn.MSELoss()
        config['train_val_split_function'] = train_val_split_regression

    ml_algorithms = {
        'SGO_IABML': SGO_IABML
    }
    print('ML algorithm = {0:s}'.format(config['ml_algorithm']))

    # Initialize a meta-learning instance
    ml = ml_algorithms[config['ml_algorithm'].capitalize()](config=config)

    if config['train_flag']:
        ml.train(train_dataloader=train_dataloader, val_dataloader=test_dataloader)
    else:
        ml.test(num_eps=config['num_episodes'], eps_dataloader=test_dataloader)