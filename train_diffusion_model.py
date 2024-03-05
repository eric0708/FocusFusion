# import packages
import os
import math
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

import torchvision.datasets as datasets
from torchvision import transforms

import matplotlib.pyplot as plt

from tqdm import tqdm

from dataset import PokemonDataset
from diffusion_model import UNet_Model

from utils import image_to_tensor_transform, tensor_to_image_transform
from utils import diffusion_forward_process
from utils import calculate_loss, generate_sample_plot, plot_training
from utils import T, img_size, num_images, step_size

def train_model(args):
    # set device
    device = args.device
    print('device is set to be {}'.format(device))

    # Define data directory path
    data_dir = args.data_dir

    if args.data_name == 'Flowers102':
        # Load dataset
        dataset = datasets.Flowers102(root=data_dir, download=True)

        # create the dataset and dataloader for the task
        train_dataset = datasets.Flowers102(root=data_dir, split='train', transform=image_to_tensor_transform, download=True)
        test_dataset = datasets.Flowers102(root=data_dir, split='test', transform=image_to_tensor_transform, download=True)
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    elif args.data_name == 'Pokemon':
        dataset = PokemonDataset(data_dir, transform=image_to_tensor_transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # check diffusion forward process results
    image_batch = next(iter(dataloader))[0]

    num_images = 10
    step_size = T // num_images
    fig, axes = plt.subplots(nrows=1, ncols=num_images+1, figsize=(15, 15))

    timestep_list = np.linspace(0, T, num_images+1).astype(int)
    timestep_list[-1] -= 1

    for idx, timestep in enumerate(timestep_list):
        t = torch.Tensor([timestep] * args.batch_size).type(torch.int64).to(device)
        noised_images, _ = diffusion_forward_process(image_batch, t)

        axes[idx].imshow(tensor_to_image_transform(noised_images[0]))
        axes[idx].set_title('t={}'.format(timestep_list[idx]))
        axes[idx].axis('off')

    # create directories and files for recording the results during training
    sample_plot_dir = args.sample_plot_dir

    if not os.path.exists(sample_plot_dir):
        os.makedirs(sample_plot_dir)

    model_checkpoint_dir = args.model_checkpoint_dir

    if not os.path.exists(model_checkpoint_dir):
        os.makedirs(model_checkpoint_dir)

    results_file = args.results_file
    results_file = open(results_file, 'w')

    # train the model
    model = nn.DataParallel(UNet_Model(use_attention=args.use_attention)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # specify checkpoint to use to resume training process
    if args.load_checkpoint:
        model.load_state_dict(torch.load(os.path.join(model_checkpoint_dir, f'epoch_{args.load_epoch:03d}_checkpoint', f'epoch_{args.load_epoch:03d}_model_checkpoint.pt')))
        optimizer.load_state_dict(torch.load(os.path.join(model_checkpoint_dir, f'epoch_{args.load_epoch:03d}_checkpoint', f'epoch_{args.load_epoch:03d}_optimizer_checkpoint.pt')))
    
    starting_epoch = args.starting_epoch
    epochs = args.epochs

    for epoch in tqdm(range(starting_epoch, epochs+1)):
        model.train()
        training_loss = 0
        for idx, (images, _) in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            timestep = torch.randint(0, T, (args.batch_size,)).to(device)
            loss = calculate_loss(model, images, timestep)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        if epoch % 50 == 0:
            if not os.path.exists(os.path.join(model_checkpoint_dir, f'epoch_{epoch:03d}_checkpoint')):
                os.makedirs(os.path.join(model_checkpoint_dir, f'epoch_{epoch:03d}_checkpoint'))
            torch.save(model.state_dict(), os.path.join(model_checkpoint_dir, f'epoch_{epoch:03d}_checkpoint', f'epoch_{epoch:03d}_model_checkpoint.pt'))
            torch.save(optimizer.state_dict(), os.path.join(model_checkpoint_dir, f'epoch_{epoch:03d}_checkpoint', f'epoch_{epoch:03d}_optimizer_checkpoint.pt'))
        
        model.eval()
        print(f'Epoch {epoch} Loss: {training_loss:.5f}')
        results_file.write(f'Epoch {epoch} Loss: {training_loss:.5f}\n')
        generate_sample_plot(model, sample_plot_dir, epoch)
    
    results_file.close()
    results_file = args.results_file
    loss_plot_file = args.loss_plot_file
    plot_training(results_file, loss_plot_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for Diffusion Model Training")
    parser.add_argument("--data_dir", type=str, default="./data/Flowers102", 
                        help="Directory containing the training dataset")
    parser.add_argument("--data_name", type=str, default="Flowers102", 
                        help="Name of the dataset used for model training")
    parser.add_argument("--sample_plot_dir", type=str, default="./sample_plots", 
                        help="Directory to save sample plots during training")
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_checkpoints", 
                        help="Directory to save model checkpoints during training")
    parser.add_argument("--results_file", type=str, default="training_results.txt", 
                        help="File to store training results")
    parser.add_argument('--loss_plot_file', type=str, default='loss_plot.png',
                        help='File name for the training loss plot')
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="Set the device used to perform the training process")
    parser.add_argument("--starting_epoch", type=int, default=1,
                        help="Number of epochs for training")
    parser.add_argument("--epochs", type=int, default=5000,
                        help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size used for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate used for the optimizer during the training process")
    parser.add_argument("--load_epoch", type=int, default=1000,
                        help="Used to specify the epoch number of the loaded model")
    parser.add_argument("--load_checkpoint", action="store_true",
                        help="Flag to indicate whether to load checkpoint or not")
    parser.add_argument("--use_attention", action="store_true",
                        help="Flag to indicate whether to use attention in the model or not")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_model(args)