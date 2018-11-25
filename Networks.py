# Adapted from https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    """
    Baseclass with helper functions for NNs
    """
    def __init__(self):
        super(Net, self).__init__()


    def train_model(self, data, target, iters=1, log_interval=100):
        self.train()
        data, target = data.to(self.device), target.to(self.device)
        for i in range(1,iters+1):
            # TODO: add functionality to use a dataloader
            #for batch_idx, (data, target) in enumerate(train_loader):
            self.optimizer.zero_grad()
            self.zero_grad()
            output = self.forward(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            if i % log_interval == 0:# or i == iters:
                print('Train Epoch: {}/{} \tLoss: {:.6f}'.format(
                    i, iters, loss.item()))
        return loss

    def test(self,test_loader,verbose=True):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            # TODO make it work with a dataloader
            for batch in test_loader:
                data, target = batch[0],batch[-1]
                data, target = data.to(self.device), target.to(self.device)
                output = self.forward(data)
                test_loss += self.loss(output, target, reduction='sum').item() # sum up batch loss
                pred = torch.argmax(output, dim=-1)#.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        acc = correct / len(test_loader.dataset)
        if verbose:
            print('\nTest set: Accuracy: {}/{} ({:.0f}%), Average loss: {:.4f}'.format(
                    correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset),
                    test_loss))

        return acc, test_loss

    def predict(self,X):
        self.eval()
        X = X.to(self.device)
        with torch.no_grad():
            out = self.forward(X)
            pred = torch.argmax(out,dim=1)
        return pred

    def acc(self,X,y):
        """
        y[i] is respective label of X[i]

        X and y must have the same size in the 0th dimension

        returns percentage of correctly labeled instances
        """
        X = X.to(self.device)
        y = y.to(self.device)
        pred = self.predict(X)
        n_correct = (pred == y).sum()
        return n_correct.item()/pred.size(0)

    def reset_parameters(self):
        def weights_init(l):
            if isinstance(l, nn.Conv2d):
                l.reset_parameters()
                #torch.nn.init.xavier_uniform_(l.weight)
            elif isinstance(l, nn.Linear):
                l.reset_parameters()
                # stdv = 1. / math.sqrt(l.weight.size(1))
                # l.weight.data.uniform_(-stdv, stdv)
                # if l.bias is not None:
                #     l.bias.data.uniform_(-stdv, stdv)

        # reset weights
        self.apply(weights_init)

        #self.apply(weights_init)
        # reset the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.lr,
                            betas=self.betas)


class LeNet(Net):
    def __init__(self,output_dim=10,img_size=32,loss=F.nll_loss,lr=0.001,
                    betas=(0.9, 0.999),device='cpu',verbose=False):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.feature_size = (((((img_size-4)//2)-4)//2)**2)*20
        #http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html
        self.fc1 = nn.Linear(self.feature_size, 50)
        self.fc2 = nn.Linear(50, output_dim)

        self.device = device
        self.output_dim = output_dim

        self.loss = loss
        self.lr = lr
        self.betas = betas

        self.verbose = verbose

        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.lr,
                             betas=self.betas)
        self.reset_parameters()

    def forward(self, x, dropout_p=0.5):
        self.conv2_drop.p = dropout_p
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.feature_size)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=dropout_p)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def dropout_uncertainty(self,x,passes=100,dropout_p=0.1):
        batch_size = x.size(0)
        out = torch.empty((passes,batch_size,self.output_dim))
        self.train()
        with torch.no_grad():
            for i in range(passes):
                pred = self.forward(x,dropout_p)
                out[i] = pred

        return out.mean(dim=0),out.std(dim=0)


class CoreSetLeNet(LeNet):
    def forward(self, x, dropout_p=0.5, return_z=False):
        self.conv2_drop.p = dropout_p
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.feature_size)
        x = self.fc1(x)
        z = F.relu(x)
        x = F.dropout(z, training=self.training, p=dropout_p)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)
        if return_z:
            return x, z
        else:
            return x


class VeNet(Net):
    def __init__(self,output_dim=10, latent_dim=50, loss=F.nll_loss, lr=0.001, betas=(0.9, 0.999),device='cpu',verbose=False):
        super(VeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        #self.fc1 = nn.Linear(500, 50)

        self.fc11 = nn.Linear(500, latent_dim)
        self.fc12 = nn.Linear(500, latent_dim)

        self.fc2 = nn.Linear(latent_dim, output_dim)

        self.device = device

        self.loss = loss
        self.lr = lr
        self.betas = betas

        self.verbose = verbose

        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.lr,
                             betas=self.betas)
        self.reset_parameters()

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu,log_var)
        out = self.classify(z)
        return out, mu, log_var

    def encode(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        mu = self.fc11(x)
        log_var = self.fc12(x)

        return mu, log_var

    def reparameterize(self, mu, logvar):
        if self.train:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def classify(self, z):
        out = self.fc2(z)
        #h3 = F.relu(self.fc3(z))
        return torch.log_softmax(out,dim=-1)

    # def decode(self,z):
    #     out = self.

    def train_model(self, data, target, iters=None, log_interval=100):
        self.train()
        data, target = data.to(self.device), target.to(self.device)
        # TODO: add functionality to use a dataloader
        #for batch_idx, (data, target) in enumerate(train_loader):
        self.optimizer.zero_grad()
        self.zero_grad()
        output,mu,log_var = self.forward(data)
        loss = self.loss(output, target)
        loss.backward()
        self.optimizer.step()

        if i % log_interval == 0:# or i == iters:
            print('Train Epoch: {}/{} \tLoss: {:.6f}'.format(
                i, iters, loss.item()))
        return loss

    def test(self,test_loader,verbose=True):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            # TODO make it work with a dataloader
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output,_,_ = self.forward(data)
                test_loss += self.loss(output, target, reduction='sum').item() # sum up batch loss
                pred = torch.argmax(output, dim=-1)#.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        acc = correct / len(test_loader.dataset)
        if verbose:
            print('\nTest set: Accuracy: {}/{} ({:.0f}%), Average loss: {:.4f}'.format(
                    correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset),
                    test_loss))

        return acc, test_loss




class SimpleNet(Net):
    def __init__(self,hidden_size=10,loss=F.nll_loss, lr=0.001, betas=(0.9, 0.999),device='cpu',verbose=False):
        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

        self.device = device
        self.loss = loss
        self.lr = lr
        self.betas = betas

        self.verbose = verbose

        self.reset_parameters()


    def forward(self, x):
        x = x.view(-1, 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# def main():
#     # Training settings
#     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#     parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                         help='number of epochs to train (default: 10)')
#     parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                         help='learning rate (default: 0.01)')
#     parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                         help='SGD momentum (default: 0.5)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                         help='how many batches to wait before logging training status')
#     args = parser.parse_args()
#     use_cuda = not args.no_cuda and torch.cuda.is_available()
#
#     torch.manual_seed(args.seed)
#
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=args.batch_size, shuffle=True, **kwargs)
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=args.test_batch_size, shuffle=True, **kwargs)
#
#
#     model = Net().to(device)
#     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#
#     for epoch in range(1, args.epochs + 1):
#         train(args, model, device, train_loader, optimizer, epoch)
#         test(args, model, device, test_loader)
#
#
# if __name__ == '__main__':
#     main()
