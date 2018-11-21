#!/usr/bin/env python3.5

import os
import random

import numpy as np
from utils.utils import save_model, load_model_dict, load_model
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.nn import functional as F

# type of GAN; 'GAN' or 'WGAN'
GAN_type = 'WGAN'
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
#image_size = 32

# Number of workers for dataloader
workers = 4

# Batch size during training
batch_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 25

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# file path for weights
generator_fp = "./models/DCGAN_generator"
discriminator_fp = "./models/DCGAN_discriminator"

datafolder = './data'

# filepath to folder with progress images None if not wanted
progess_imgs_folder = './progress_imgs/'

# Establish convention for real and fake labels during training
real_label = 1
fake_label = -1

# max amount of gpus used for training
ngpu = 1

# select GPU if available
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


# Generator Code
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
        )
        self.main2 = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )
        self.apply(weights_init)

    def forward(self, input):
        #print('Gen input: ', input.shape)
        out = self.main(input)
        #print('Gen middle: ', out.shape)
        out = self.main2(out)
        #print('Gen output: ', out.shape)
        return out


# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),)
        self.main2 = nn.Sequential(
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # 8, 1, 4, 1, 0, bias=False),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, input):
        #print('Disc input', input.shape)
        out = self.main(input)
        # print(out.shape)
        out = self.main2(out)
        #print('Disc out: ', out.shape)
        return out

# Critic Code (same as discriminator w/o sigmoid)
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),)
        self.main2 = nn.Sequential(
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # 8, 1, 4, 1, 0, bias=False),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )
        self.apply(weights_init)

    def forward(self, input):
        #print('Disc input', input.shape)
        out = self.main(input)
        # print(out.shape)
        out = self.main2(out)
        #print('Disc out: ', out.shape)
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(batch, netG, netD, criterion, optimizerD, optimizerG):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # Train with all-real batch
    netD.zero_grad()
    # Format batch
    real_cpu = batch.to(device)
    b_size = real_cpu.size(0)
    label = torch.full((b_size,), real_label, device=device)

    # Forward pass real batch through D
    output = netD(real_cpu).view(-1)
    # Calculate loss on all-real batch
    errD_real = criterion(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()

    # Train with all-fake batch
    # Generate batch of latent vectors
    noise = torch.randn(b_size, nz, 1, 1, device=device)
    # Generate fake image batch with G
    fake = netG(noise)
    label.fill_(fake_label)
    # Classify all fake batch with D
    # print('Generating',fake.detach().shape)
    output = netD(fake.detach()).view(-1)

    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)
    # Calculate the gradients for this batch
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    # Add the gradients from the all-real and all-fake batches
    errD = errD_real + errD_fake
    # Update D
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = netD(fake).view(-1)

    # Calculate G's loss based on this output
    errG = criterion(output, label)
    # Calculate gradients for G
    errG.backward()
    D_G_z2 = output.mean().item()
    # Update G
    optimizerG.step()
    return errG.item(), errD.item(), D_x, D_G_z1, D_G_z2

def train_wasserstein_C(batch, netG, netC, optimizerC, c = 0.01):
    # https://arxiv.org/pdf/1701.07875.pdf
    for p in netC.parameters(): # reset requires_grad
        p.requires_grad = True # they are set to False below in netG update
    netC.zero_grad()
    netG.zero_grad()

    ########## Update critic ############
    # sample new batch
    batch_size = batch.size(0)

    # sample noise
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    C_x = netC(batch).mean()
    C_G_z = netC(netG(noise)).mean()
    lossC = -(C_x - C_G_z)
    lossC.backward()
    optimizerC.step()

    # Weight clipping

    for p in netC.parameters():
        p.data.clamp_(-c, c)

    return lossC.item(), C_x.item(), C_G_z.item()

def train_wasserstein_G(netG, netC, optimizerG, batch_size=batch_size):
    for p in netC.parameters():
        p.requires_grad = False # to avoid computation

    netG.zero_grad()
    ########## Update Generator ###########
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    C_G_z = netC(netG(noise)).mean()
    lossG = -C_G_z
    lossG.backward()
    optimizerG.step()

    return lossG.item(), C_G_z.item()

"""--------------------------- Train script -------------------------------"""

if __name__ == '__main__':

    manual_seed = 1338
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    print(torch.__version__)

    ############ Utility Functions ############
    def save_progress_img(fp, generator, noise, showplot=False):
        '''
        creates an image of the images generated by generator(noise)
        and saves it at fp
        '''
        with torch.no_grad():
            fake = generator(noise).detach().cpu()
        img = vutils.make_grid(fake, padding=2, normalize=True)
        plt.figure(figsize=(8,8))
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.axis('off')
        plt.savefig(fp)
        if showplot:
            plt.show()

    ############ building Dataset #################
    trans = transforms.Compose([transforms.Pad(2),  # pad MNIST images to 32x32
                                transforms.ToTensor(),
                                # shift values to [-1,1] for tanh
                                transforms.Normalize((.5,), (.5,)),
                                ])
    # download mnist dataset
    train_set = dset.MNIST(root=datafolder, train=True,
                           transform=trans, download=True)
    test_set = dset.MNIST(root=datafolder, train=False,
                          transform=trans, download=True)

    dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)


    if GAN_type == 'GAN':
        # Create the generator
        netG = Generator().to(device)

        # Create the Discriminator
        netD = Discriminator().to(device)

        # if (device.type == 'cuda') and (ngpu > 1):
        #    netG = nn.DataParallel(netG, list(range(ngpu)))
        #    netD = nn.DataParallel(netD, list(range(ngpu)))

        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


        iters = 0
        G_losses, D_losses = [],[]

        epochD = 0
        epochG = 0

        # Loading weights from file
        dictG = load_model_dict(generator_fp + '.pth.tar', map_location=device)
        if dictG is not None:
            G_losses, epochD = load_model(netG, dictG, optimizer=optimizerG)

        dictD = load_model_dict(discriminator_fp + '.pth.tar', map_location=device)
        if dictD is not None:
            D_losses, epochG = load_model(netD, dictD, optimizer=optimizerD)

        print("Starting Training Loop...")
        for epoch in range(num_epochs):
            for i, (batch_X, batch_y) in enumerate(dataloader, 0):
                iters += 1
                errG, errD, D_x, D_G_z1, D_G_z2 = train(batch_X, netG, netD,
                                                        criterion, optimizerD,
                                                        optimizerG)

                # Save Losses for plotting later
                G_losses.append(errG)
                D_losses.append(errD)

                # Output training stats
                if i % 1 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(dataloader),
                             errD, errG, D_x, D_G_z1, D_G_z2))

            # Save progress image of generator by saving G's output on fixed_noise every epoch
            if progess_imgs_folder is not None:
                fp_img = progess_imgs_folder+'epoch-{}.jpg'.format(epoch)
                save_progress_img(fp_img, netG, fixed_noise, showplot=False)

            epochD += 1
            epochG += 1

            # Save model
            fp_G = generator_fp + '{}.pth.tar'.format(epoch)
            save_model(fp_G, netG, optimizerG, G_losses, epoch)

            fp_D = discriminator_fp + '{}.pth.tar'.format(epoch)
            save_model(fp_D, netD, optimizerD, D_losses, epoch)

    elif GAN_type == 'WGAN':
        lr = 0.00005
        c = 0.01
        n_critic = 5

        # Create the generator
        netG = Generator().to(device)

        # Create the Discriminator
        netC = Critic().to(device)

        # if (device.type == 'cuda') and (ngpu > 1):
        #    netG = nn.DataParallel(netG, list(range(ngpu)))
        #    netC = nn.DataParallel(netD, list(range(ngpu)))


        # Setup Adam optimizers for both G and D
        optimizerC = optim.RMSprop(netC.parameters(), lr=lr)
        optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

        iters = 0
        G_losses, C_losses = [],[]

        epochC = 0
        epochG = 0

        # Loading weights from file
        dictG = load_model_dict(generator_fp + '.pth.tar', map_location=device)
        if dictG is not None:
            G_losses, epochD = load_model(netG, dictG, optimizer=optimizerG)

        dictC = load_model_dict(discriminator_fp + '.pth.tar', map_location=device)
        if dictC is not None:
            C_losses, epochG = load_model(netD, dictD, optimizer=optimizerD)

        print("Starting Training Loop...")

        for epoch in range(num_epochs):
            for i, (batch,_) in enumerate(dataloader):
                batch = batch.to(device)
                lossC, C_x, C_G_z1 = train_wasserstein_C(batch, netG, netC, optimizerC, c=c)
                C_losses.append(lossC)
                if (i % n_critic) == 0:
                    # only train generator every n_critic steps
                    lossG, C_G_z2 = train_wasserstein_G(netG, netC, optimizerG, batch_size=batch.size(0))
                    G_losses.append(lossG)

                # Output training stats
                if i % 1 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(dataloader),
                             lossC, lossG, C_x, C_G_z1, C_G_z2))

            if progess_imgs_folder is not None:
                fp_img = progess_imgs_folder+'-{}.jpg'
                save_progress_img(fp_img, netG, fixed_noise, showplot=False)

            epochC += 1
            epochG += 1

            # Save model
            fp_G = generator_fp + '{}-WGAN.pth.tar'.format(epoch)
            save_model(fp_G, netG, optimizerG, G_losses, epochG)

            fp_C = discriminator_fp + '{}-WGAN.pth.tar'.format(epoch)
            save_model(fp_C, netC, optimizerC, C_losses, epochC)
