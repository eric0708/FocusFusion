# import packages
import os
import numpy as np
import matplotlib.pyplot as plt

import PIL.Image as Image

import torch
import torch.nn.functional as F

from torchvision import transforms

from ignite.metrics import FID, InceptionScore
from ignite.engine import Engine

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define some parameters
img_size = 64
num_images = 10
step_size = 30

# define parameters used in diffusion forward process
beta_0 = 0.0001
T = 300
beta_T = beta_0 * T 

# calculate terms used in closed form notation of sampling at different timestamps
betas = torch.linspace(beta_0, beta_T, T).to(device)
alphas = 1 - betas
cum_prod_alphas = torch.cumprod(alphas, dim=0)
prev_cum_prod_alphas = F.pad(cum_prod_alphas[:-1], (1, 0), value=1.0)
sqrt_cum_prod_alphas = torch.sqrt(cum_prod_alphas)
sqrt_one_minus_cum_prod_alphas = torch.sqrt(1 - cum_prod_alphas)
sqrt_one_over_alphas = 1 / torch.sqrt(alphas)
standard_deviations = torch.sqrt((1 - prev_cum_prod_alphas) / (1 - cum_prod_alphas) * betas)

# define some transformations to be applied while loading the dataset
image_to_tensor_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (2 * t) - 1)
])

# pipeline for transforming tensors back to image
tensor_to_image_transform = transforms.Compose([
    transforms.Lambda(lambda t: 255 * (t + 1) / 2),
    transforms.Lambda(lambda t: t.permute(1, 2, 0)),
    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
    transforms.ToPILImage()
])

# get corresponding index entries from tensors
def get_from_index(tensor_t, timestep):
    return tensor_t.gather(dim=0, index=timestep).reshape(timestep.shape[0], 1, 1, 1)

# define diffusion forward process
def diffusion_forward_process(x_0, timestep):
    noise = torch.randn_like(x_0).to(device)
    sqrt_cum_prod_alphas_t = get_from_index(sqrt_cum_prod_alphas, timestep)
    sqrt_one_minus_cum_prod_alphas_t = get_from_index(sqrt_one_minus_cum_prod_alphas, timestep)
    noised_images = sqrt_cum_prod_alphas_t.to(device) * x_0.to(device) + sqrt_one_minus_cum_prod_alphas_t.to(device) * noise

    return noised_images, noise

# define function for calculating the loss between actual noise and predicted noise
def calculate_loss(model, x_0, timestep):
    noised_images, noise = diffusion_forward_process(x_0, timestep)
    noise_pred = model(noised_images, timestep)
    loss = F.l1_loss(noise, noise_pred)
    return loss

# define function for denoising an image given the noisy image and the timestep
def denoise_image(model, x, timestep):
    model.eval()
    with torch.no_grad():
        betas_t = get_from_index(betas, timestep)
        sqrt_one_minus_cum_prod_alphas_t = get_from_index(sqrt_one_minus_cum_prod_alphas, timestep)
        sqrt_one_over_alphas_t = get_from_index(sqrt_one_over_alphas, timestep)
        noise_pred = model(x, timestep)
        image_mean = sqrt_one_over_alphas_t * (x - betas_t / sqrt_one_minus_cum_prod_alphas_t * noise_pred)
        
        if timestep == 0:
            return image_mean
        else:
            standard_deviations_t = get_from_index(standard_deviations, timestep)
            noise = torch.randn_like(x).to(device)
            return image_mean + standard_deviations_t * noise      

# define function for generating the process of denoising a sample 
def generate_sample_plot(model, sample_plot_dir, epoch):
    model.eval()
    with torch.no_grad():
        noised_image = torch.randn((1, 3, img_size, img_size)).to(device)

        fig, axes = plt.subplots(nrows=1, ncols=num_images+1, figsize=(35, 3))
        plot_counter = 0

        for timestep in reversed(range(0, T)):
            timestep_tensor = torch.Tensor([timestep]).type(torch.int64).to(device)
            noised_image = denoise_image(model, noised_image, timestep_tensor)
            
            if timestep % step_size == 0 or timestep == T - 1:
                axes[plot_counter].imshow(tensor_to_image_transform(noised_image[0]))
                axes[plot_counter].set_title('t={}'.format(timestep))
                axes[plot_counter].axis('off')
                plot_counter += 1
        plt.savefig(os.path.join(sample_plot_dir, f'epoch_{epoch:03d}_sample_plot.png'))
        plt.show() 

# define transformation to resize image in order to be used by ignite
resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

# define function to interpolate images in order to be used by ignite
def interpolate(image_batch):
    transformed_image = []
    for image in image_batch:
        transformed_image.append(resize_transform(image))
    return torch.stack(transformed_image)

# define function for plotting the training loss curve
def plot_training(results_file, loss_plot_file):
    train_loss_list = []

    with open(results_file, 'r') as f:
        for line in f:
            train_loss_list.append(float(line.split()[-1]))

    epoch_list = [epoch_num+1 for epoch_num in range(len(train_loss_list))]

    plt.plot(epoch_list, train_loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('L1 Train Loss')
    plt.title('Diffusion Model L1 Train Loss Against Epochs (T=300)')
    plt.savefig(loss_plot_file)
    plt.show()

# set global model variable
global model

# define evaluation step to generate images and interpolate generated and real images
def evaluation_step(engine, data_batch):
    image_batch, _ = data_batch

    with torch.no_grad():
        noised_image = torch.randn((image_batch.shape[0], 3, img_size, img_size)).to(device)

        for timestep in reversed(range(0, T)):
            timestep_tensor = torch.Tensor([timestep]).type(torch.int64).to(device)
            noised_image = denoise_image(model, noised_image, timestep_tensor)

    generated_images = interpolate(noised_image)
    real_images = interpolate(image_batch)

    return generated_images, real_images

# define some variables for calculating FID and IS
fid = FID(device=device)
inception = InceptionScore(device=device, output_transform=lambda x: x[0])
evaluator = Engine(evaluation_step)
fid.attach(evaluator, "fid")
inception.attach(evaluator, "inception")

# define evaluation function to calculate FID and IS
def evaluate(evaluated_model, dataloader):
    global model
    model = evaluated_model
    model.to(device)
    model.eval()
    evaluator.run(dataloader, max_epochs=1)
    metrics = evaluator.state.metrics
    fid_score = metrics['fid']
    is_score = metrics['inception']
    
    return fid_score, is_score
