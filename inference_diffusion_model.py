# import packages
import os
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from dataset import PokemonDataset
from diffusion_model import UNet_Model

from utils import image_to_tensor_transform, tensor_to_image_transform
from utils import denoise_image, evaluate
from utils import device, T, img_size

def inference_model(args):
    # set device
    device = args.device
    print('device is set to be {}'.format(device))

    # define batch size
    batch_size = args.batch_size

    # Define data directory path
    data_dir = args.data_dir

    if args.data_name == 'Flowers102':
        # create the dataset for calculating FID and IS scores
        train_dataset = datasets.Flowers102(root=data_dir, split='train', transform=image_to_tensor_transform, download=True)
        test_dataset = datasets.Flowers102(root=data_dir, split='test', transform=image_to_tensor_transform, download=True)
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    elif args.data_name == 'Pokemon':
        dataset = PokemonDataset(data_dir, transform=image_to_tensor_transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # define file paths
    model_checkpoint_dir = args.model_checkpoint_dir

    # initiate and load the model
    checkpoint_epoch = args.checkpoint_epoch
    model = nn.DataParallel(UNet_Model(use_attention=args.use_attention)).to(device)
    model.load_state_dict(torch.load(os.path.join(model_checkpoint_dir, f'epoch_{checkpoint_epoch:03d}_checkpoint', f'epoch_{checkpoint_epoch:03d}_model_checkpoint.pt')))

    # generate example plots using the model
    nrows = args.nrows
    ncols = args.ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))

    noised_image = torch.randn((nrows * ncols, 3, img_size, img_size)).to(device)

    for timestep in reversed(range(0, T)):
        timestep_tensor = torch.Tensor([timestep]).type(torch.int64).to(device)
        noised_image = denoise_image(model, noised_image, timestep_tensor)

    for idx in range(nrows * ncols):
        row_idx = idx // ncols
        col_idx = idx % ncols

        axes[row_idx][col_idx].imshow(tensor_to_image_transform(noised_image[idx]))
        axes[row_idx][col_idx].axis('off')

    plt.savefig(args.sample_plot_file)
    plt.show() 

    fid_score, is_score = evaluate(model, dataloader)
    print(f'FID: {fid_score:.5f}, IS: {is_score:.5f}')

    results_file = args.results_file
    results_file = open(results_file, 'w')
    results_file.write(f'FID: {fid_score:.5f}, IS: {is_score:.5f}')

def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for Inference")

    parser.add_argument('--data_dir', type=str, default='./data/Flowers102',
                        help='Directory containing the dataset')
    parser.add_argument('--data_name', type=str, default='Flowers102',
                        help='Name of the dataset')
    parser.add_argument('--model_checkpoint_dir', type=str, default='./model_checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="Set the device used to perform the inference process")
    parser.add_argument('--checkpoint_epoch', type=int, default=5000,
                        help='Epoch at which to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size used for model inference')
    parser.add_argument('--nrows', type=int, default=8,
                        help='Number of rows in the sample plot')
    parser.add_argument('--ncols', type=int, default=8,
                        help='Number of columns in the sample plot')
    parser.add_argument('--sample_plot_file', type=str, default='sample_plots.png',
                        help='File name for the sample plot')
    parser.add_argument('--results_file', type=str, default='inference_results.txt',
                        help='File name for the inference results')
    parser.add_argument("--use_attention", action="store_true",
                        help="Flag to indicate whether to use attention in the model or not")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    inference_model(args)