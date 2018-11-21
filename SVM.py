import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


# Simple linear SVM  with SGD in PyTorch
class SVM(nn.Module):
    def __init__(self,features=32*32,classes=2,lr=0.01, betas=(0.9, 0.999), device='cpu', verbose=False):
        super(SVM, self).__init__()
        self.fc = nn.Linear(features,classes)
        # initialize criterion
        self.criterion = nn.NLLLoss()

        # initialize optimizer
        # Reminder for future changes:
        # !!! optimizer gets reset to adam in reset_parameters !!!
        self.lr = lr
        self.betas = betas
        self.optimizer = optim.Adam(self.parameters(),
                               lr=self.lr, betas=self.betas)

        self.device = device
        self = self.to(device)

        self.verbose = verbose

    def forward(self,x):
        x = x.view(x.size()[0],-1)
        out = self.fc(x)
        out = F.log_softmax(out,dim=1)
        return out

    def train_model(self,X,y,iterations=100,
              X_test=None,y_test=None):
        self.train()
        X = X.to(self.device)
        y = y.to(self.device)

        # begin training loop
        for i in range(1,iterations+1):
            self.zero_grad()
            self.optimizer.zero_grad()

            out = self.forward(X)#.view(-1)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            if self.verbose and (i % 100 == 0 or i == iterations):
                if X_test is not None and y_test is not None:
                    print('Iteration: {} | Loss: {} | Test acc: {}'.format(
                        i,loss,self.acc(X_test,y_test)))
                else:
                    print('Iteration: {} | Loss: {}'.format(i,loss))

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

    def test(self, test_loader, device='cpu'):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                out = self.forward(data)
                test_loss += self.criterion(out, target)
                pred = torch.argmax(out,dim=1)
                n_correct = (pred == target).sum()
                correct += n_correct

        avg_loss = test_loss / len(test_loader.dataset)
        acc = correct.float() / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avg_loss, correct, len(test_loader.dataset),acc*100))
        return avg_loss, acc

    def reset_parameters(self):
        self.fc.reset_parameters()
        self.optimizer = self.optimizer = optim.Adam(self.parameters(),
                               lr=self.lr, betas=self.betas)



# import numpy as np
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# # Simple linear SVM  with SGD in PyTorch
# class SVM(nn.Module):
#     def __init__(self,features=32*32,classes=2,lr=0.01, beta1=0.5):
#         super(SVM, self).__init__()
#         self.fc = nn.Linear(features,classes)
#         # initialize criterion
#         self.criterion = nn.CrossEntropyLoss()
#
#         # initialize optimizer
#         self.optimizer = optim.Adam(self.parameters(),
#                                lr=lr, betas=(beta1, 0.999))
#
#         # TODO: initialize weights
#
#     def forward(self,x):
#         x = x.view(x.size()[0],-1)
#         out = self.fc(x)
#         return out
#
#     def train_model(self,X,y,iterations=1000,
#               X_test=None,y_test=None):
#         self.train()
#
#         # begin training loop
#         for i in range(1,iterations+1):
#             self.zero_grad()
#             out = self.forward(X)#.view(-1)
#             loss = self.criterion(out, # output of network
#                              y#.float() # targets
#                             )
#             loss.backward()
#             self.optimizer.step()
#
#             if i % 100 == 0 or i == iterations:
#                 if X_test is not None and y_test is not None:
#                     print('Iteration: {} | Loss: {} | Test acc: {}'.format(
#                         i,loss,self.acc(X_test,y_test)))
#                 else:
#                     print('Iteration: {} | Loss: {}'.format(i,loss))
#
#     def predict(self,X):
#         self.eval()
#         #with torch.no_grad():
#         out = self.forward(X)#.view(-1)
#         out[out <= 0] = -1
#         out[out > 0] = 1
#         return out
#
#     def acc(self,X,y):
#         """
#         y[i] is respective label of X[i]
#         y is either -1 or 1
#         X and y must have the same size in the 0th dimension
#
#         returns percentage of correctly labeled instances
#         """
#         pred = self.predict(X)
#         n_correct = (pred == y).sum()
#         return n_correct.item()/pred.size()[0]
#
#     def test(self, test_loader, device='cpu'):
#         self.eval()
#         test_loss = 0
#         correct = 0
#         with torch.no_grad():
#             for data, target in test_loader:
#                 data, target = data.to(device), (target).to(device) #*2-1
#                 pred = self.predict(data)
#
#
#                 out = self.forward(data)
#                 test_loss += self.criterion(out,#.view(-1), # output of network
#                                  target) # targets
#                 n_correct = (torch.argmax(pred) == target).sum()  #.float()
#                 correct += n_correct
#
#
#         avg_loss = test_loss / len(test_loader.dataset)
#         acc = correct.float() / len(test_loader.dataset)
#         print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#             avg_loss, correct, len(test_loader.dataset),acc*100))
#         return avg_loss, acc
