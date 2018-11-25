import os
import random

import utils.utils as utils

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
#import torch.utils.data
#import torchvision.datasets as dset
#import torchvision.transforms as transforms
#import torchvision.utils as vutils
from torch.nn import functional as F


class ActiveLearner(object):
    """Base class for Activer Learners"""

    def __init__(self, model, criterion, dataset, L_indices, val_loader=None, verbose=False, device='cpu'):
        """
        model: the SVM that will be trained
        criterion: criterion that measures the usefullness of an
                unlabeled instance
        L_x,L_y: initially labeled data pool
        U_x,U_y: unlabeled data pool
        """
        self.model = model
        self.criterion = criterion
        self.dataset = dataset
        self.val_loader = val_loader
        if val_loader is None:
            print('Warning: no val_loader!')

        #mask = np.isin(np.arange(len(dataset)),np.array(L_indices,dtype=int)) == False
        #U_indices = np.arange(len(dataset))[mask]

        # np.array(L_indices,dtype=int))
        self.L = torch.utils.data.Subset(self.dataset, np.array([],dtype=int))
        self.U = torch.utils.data.Subset(
            self.dataset, np.arange(len(dataset)))  # [U_indices])
        self.label(L_indices)

        self.device = device
        self.verbose = verbose

    def label(self, i):
        """
        i: int or iterable of int
        label instances at U_x[i] and remove it from the U and put it in L
        TODO: implement batch sampling
        """
        if type(i) == int:
            i = [i]
        if len(i) == 0:
            return

        # index of instance_i in the underlying (whole) dataset
        X_index = self.U.indices[i]
        # print(X_index)
        self.U.indices = np.delete(self.U.indices, i)
        self.L.indices = np.append(self.L.indices, X_index)

        if hasattr(self.criterion, 'update_cache'):
            self.criterion.update_cache(i)
            assert(len(self.U) == len(self.criterion.U_z))
            print('Updating criterion cache')

    def re_train(self, epochs=1, batch_size=32, hard=True):
        #print("Please check if correct")
        if hard:
            self.model.reset_parameters()

        if self.verbose:
            print('Retraining model with {} datapoints'.format(len(self.L)))

        kwargs = {'num_workers': 1,
                  'pin_memory': True} if torch.cuda.is_available() else {'num_workers': 2}

        if not len(self.L) // batch_size >= 10:
            batch_size = max(1, len(self.L) // 10)

        L_dataloader = torch.utils.data.DataLoader(
            self.L, batch_size=batch_size, shuffle=True, **kwargs)



        val_losses = []
        val_accs = []
        for epoch in range(1, epochs + 1):
            loss = 0
            batches = 0
            self.model.train()
            for batch in L_dataloader:
                X = batch[0]
                y = batch[-1]
                batches += 1
                loss += self.model.train_model(X, y, iters=1)
            avg_loss = loss / batches

            #validate
            if self.val_loader is not None:
                val_acc,val_loss = self.model.test(self.val_loader,verbose=False)
                # TODO: implement sophisticated early stopping
                # if len(val_losses) >=5 and val_loss>val_losses[-1]:
                #
                #     print('Overfitting detected early_stopping at epoch {} prev/current loss {:.4f}/{:.4f} | acc {:.3f}/{:.3f}'.format(epoch,val_losses[-1],val_loss,val_accs[-1],val_acc))
                #
                #     #early Stopping
                #     break

                if len(val_losses)==0 or val_loss < min(val_losses):
                    # create checkpoint for best model so far to revert to in
                    # case of overfitting

                    #print('New best val_loss {:.4f} | Acc {:.4f}'.format(val_loss,val_acc))
                    state = {
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.model.optimizer.state_dict(),
                    }

                    torch.save(state, 'temp_best_checkpoint.pth')


                val_losses.append(val_loss)
                val_accs.append(val_acc)

        # reload best model by val loss
        if self.val_loader is not None:
            #print('Reverting to best model...')
            state = torch.load('temp_best_checkpoint.pth')
            self.model.load_state_dict(state['state_dict'], strict=True)
            self.model.optimizer.load_state_dict(state['optimizer'])
            val_acc,val_loss = self.model.test(self.val_loader,verbose=False)
            print('Best val_loss {:.4f} | Acc {:.4f}'.format(val_loss,val_acc))
            # sanity check
            assert(min(val_losses) == val_loss)


        if self.verbose:
            print("Epoch {} | Avg Loss: {:.4f}".format(epoch, avg_loss))

    def plot(self, densfunc, resolution=150, show_criterion=False):
        """
        show_criterion currently only works if real data is 2-dimensional
        """
        plt.figure(figsize=(6, 6))

        if show_criterion:
            # Sample data
            side = np.linspace(-7, 7, resolution)
            X_dens, Y_dens = np.meshgrid(side, side)
            zipped = np.dstack((X_dens, Y_dens)).reshape(-1, 2)
            dens = torch.tensor(densfunc(torch.tensor(
                zipped, dtype=torch.double)), dtype=torch.float)

            self.model.train
            with torch.no_grad():
                pred = self.model.forward(
                    torch.tensor(zipped, dtype=torch.float))
            Z_dens = self.criterion(
                pred,
                dens
            ).reshape(resolution, resolution)

            # Plot the density map using nearest-neighbor interpolation

            plt.pcolormesh(X_dens, Y_dens, Z_dens, cmap='Greys')

        # plot unlabeled points
        X = self.U.dataset[self.U.indices][1]
        y = self.U.dataset[self.U.indices][-1]
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y,
                        s=5, alpha=0.2, legend=False)

        # plot labeled points
        X = self.L.dataset[self.L.indices][1]
        y = self.L.dataset[self.L.indices][-1]
        sns.scatterplot(x=X[:, 0], y=X[:, 1], s=50, legend=False, hue=y)

        plt.show()

    def select(self, temp=0):
        """
        Selects the next instance or batch to label
        returns its index in self.U_x
        """

        scores = self.calc_scores()

        # make array of pseudo probability distribution
        p = scores / scores.sum()

        i = self.sample(p, temp=temp)

        if self.verbose:
            print("Selected instance {} with score {:.4f}".format(
                i, scores[i]))

        return i

    def calc_scores(self):
        """
        calculates the utility score of each instance in self.U given by criterion
        """
        pass

    def sample(self, p, temp):
        """
        sample an index of an instance given probability distribution p and
        temperature temp
        """

        if temp == 0:
            # lim temp -> 0 samples argmax(p) with prob 1
            i = torch.argmax(p).item()
        else:

            # if p.sum().item() > 1.0 + 1e-18 or p.sum().item() < 1.0 - 1e-18:
            #     print('not a probability distribution',p.sum())
            #     p /= p.sum().item()

            if temp != 1:
                p_temp = torch.exp(torch.log(p) / temp)
                p_temp /= p_temp.sum()
                # print(p_temp.sum())
            else:
                # temp == 1 does nothing
                p_temp = p

            # sample index from distribution given by p_temp
            i = np.random.choice(p.shape[0], p=p_temp.numpy())

            if self.verbose:
                print("Selected instance {} with p {:.4f} | p_temp {:.4f}".format(
                    i, p[i], p_temp[i]))

        return i


class ModelAgnosticActiveLearner(ActiveLearner):
    def calc_scores(self):
        """
        calculates the utility score of each instance in self.U given by criterion

        TODO: implement top_k for batch sampling
        """
        if torch.cuda.is_available():
            kwargs = {'num_workers': 1, 'pin_memory': True}
        else:
            kwargs = {}
        U_dataloader = torch.utils.data.DataLoader(
            self.U, batch_size=len(self.U), shuffle=False, **kwargs)

        scores = None
        for data in U_dataloader:
            batch_scores = self.criterion(*data[0:-1])
            if scores is None:  # first iteration
                scores = batch_scores
            else:
                scores = torch.cat((scores, batch_scores), 0)

        return scores


class RandomActiveLearner(ActiveLearner):
    """
    query a random instance every iteration = passive learning
    """

    def select(self, temp=None):
        """
        overwriting select function of ActiveLearner
        """
        i = random.randint(0, len(self.U) - 1)
        return i


class ModelAgnosticActiveLearner(ActiveLearner):
    def calc_scores(self):
        """
        calculates the utility score of each instance in self.U given by criterion

        TODO: implement top_k for batch sampling
        """
        if torch.cuda.is_available():
            kwargs = {'num_workers': 1, 'pin_memory': True}
        else:
            kwargs = {}
        U_dataloader = torch.utils.data.DataLoader(
            self.U, batch_size=len(self.U), shuffle=False, **kwargs)

        L_z = self.L.dataset[self.L.indices][1]
        scores = None
        for X, z, _ in U_dataloader:  # data = [X,z,'y']
            U_z = z
            batch_scores = self.criterion(None, U_z, L_z)
            if scores is None:  # first iteration
                scores = batch_scores
            else:
                scores = torch.cat((scores, batch_scores), 0)

        return scores


class SearchActiveLearner(ActiveLearner):
    """
    Linear search for best instance according to criterion
    dataset[i] = x,y of datapoint i
    """

    def __init__(self, model, criterion, dataset, L_indices, val_loader=None,verbose=False, device='cpu'):
        super().__init__(model, criterion, dataset, L_indices, val_loader=val_loader,
                         verbose=verbose, device=device)

    def calc_scores(self):
        """
        calculates the utility score of each instance in self.U given by criterion

        TODO: implement top_k for batch sampling
        """

        if torch.cuda.is_available():
            kwargs = {'num_workers': 1, 'pin_memory': True}
        else:
            kwargs = {}
        U_dataloader = torch.utils.data.DataLoader(
            self.U, batch_size=len(self.U), shuffle=False, **kwargs)

        self.model.eval()

        with torch.no_grad():
            scores = None
            for data in U_dataloader:
                X = data[0]
                pred = self.model(X)
                batch_scores = self.criterion(pred, *data[1:-1])
                if scores is None:  # first iteration
                    scores = batch_scores
                else:
                    scores = torch.cat((scores, batch_scores), 0)

        return scores

class CoreSetActiveLearner(ActiveLearner):
    """
    Linear search for best instance according to criterion
    dataset[i] = x,y of datapoint i

    based on https://arxiv.org/abs/1708.00489
    """

    def __init__(self, model, criterion, dataset, L_indices, val_loader=None, verbose=False, device='cpu'):
        super().__init__(model, criterion, dataset, L_indices, val_loader=val_loader,
                         verbose=verbose, device=device)

    def calc_scores(self):
        """
        calculates the utility score of each instance in self.U given by criterion
        """

        if torch.cuda.is_available():
            kwargs = {'num_workers': 1, 'pin_memory': True}
        else:
            kwargs = {}
        U_dataloader = torch.utils.data.DataLoader(
            self.U, batch_size=len(self.U), shuffle=False, **kwargs)
        L_dataloader = torch.utils.data.DataLoader(
            self.L, batch_size=len(self.L), shuffle=False, **kwargs)

        L_x = next(iter(L_dataloader))[0]

        self.model.eval()

        with torch.no_grad():
            _,L_z = self.model.forward(L_x,return_z=True)
            scores = None
            for data in U_dataloader:
                X = data[0]
                pred,U_z = self.model.forward(X,return_z=True)
                batch_scores = self.criterion(pred, U_z=U_z,L_z=L_z)
                if scores is None:  # first iteration
                    scores = batch_scores
                else:
                    scores = torch.cat((scores, batch_scores), 0)

        return scores

class DiverstiyDensityUncertaintyActiveLearner(ActiveLearner):
    """
    Linear search for best instance according to criterion using the latent
    maping given in the dataset
    dataset[i] = x,z,y of datapoint i
    """

    def __init__(self, model, criterion, dataset, L_indices, val_loader=None, verbose=False, device='cpu'):
        super().__init__(model,
                         criterion,
                         dataset,
                         L_indices,
                         val_loader=val_loader,
                         verbose=verbose,
                         device=device)

        self.lambda_ = 0

    def calc_scores(self):
        """
        returns argmax_(x in U_x) criterion(model(x),z)
        TODO: implement top_k for batch sampling
        TODO: implement batchwise score calculation but with golbal normalization
        """

        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
        }
        U_dataloader = torch.utils.data.DataLoader(
            self.U, batch_size=len(self.U), shuffle=False, *kwargs)

        self.model.eval()

        with torch.no_grad():
            L_z = self.L.dataset[self.L.indices][1]
            scores = None
            for X, z, _ in U_dataloader:  # data = [X,z,'y']
                U_z = z
                pred = self.model.forward(X)
                batch_scores = self.criterion(pred, U_z, L_z)
                if scores is None:  # first iteration
                    scores = batch_scores
                else:
                    scores = torch.cat((scores, batch_scores), 0)

        return scores

    def plot(self, densfunc, resolution=150, show_criterion=False):
        """
        show_criterion currently only works if real data is 2-dimensional
        """
        plt.figure(figsize=(6, 6))
        if show_criterion:
            # Sample data
            side = np.linspace(-7, 7, resolution)
            X_dens, Y_dens = np.meshgrid(side, side)
            zipped = np.dstack((X_dens, Y_dens)).reshape(-1, 2)

            self.model.eval()
            with torch.no_grad():
                pred = self.model.forward(
                    torch.tensor(zipped, dtype=torch.float))
            Z_dens = self.criterion(
                pred,
                torch.tensor(zipped, dtype=torch.float),
                self.L.dataset[self.L.indices][1],
                lambda_=self.lambda_
            ).reshape(resolution, resolution)

            # Plot the density map using nearest-neighbor interpolation
            plt.pcolormesh(X_dens, Y_dens, Z_dens, cmap='Greys')

        # plot unlabeled points
        X = self.U.dataset[self.U.indices][1]
        y = self.U.dataset[self.U.indices][-1]
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y,
                        s=5, alpha=0.2, legend=False)

        # plot labeled points
        X = self.L.dataset[self.L.indices][1]
        y = self.L.dataset[self.L.indices][-1]
        sns.scatterplot(x=X[:, 0], y=X[:, 1], s=50, legend=False, hue=y)

        plt.show()

# ---------------- OLD CODE NEEDS REFACTORING ------------------
# class LatentOptimizationActiveLearner(ActiveLearner):
#     """
#     Continious optimization to maximize criterion w.r.t. latent variables of a generator
#     """
#     def __init__(self,model,criterion,L_x,L_y,U_x,U_y,generator,nz,verbose=False,device='cpu'):
#         """
#         L_x,L_y,U_x,U_y: see Active Learner
#         generator: generator model
#         nz: latent size of generator
#         criterion: z will be optimized to maximize criterion
#         """
#         super().__init__(model,criterion,L_x,L_y,U_x,U_y,verbose=verbose,device=device)
#         self.generator = generator
#         self.nz = nz
#
#     def select(self):
#         # sample standard normal input noise that tracks gradient
#         k = 20
#         z = torch.randn(k, self.nz, 1, 1, device=self.device,requires_grad=True)
#         # optimize k zs
#         z = self.optimize(z, iterations=25, lr=0.05)
#
#
#         # select z with highest entropy of G(z)
#         z = z.detach()
#         with torch.no_grad():
#             gen_inst = self.generator.forward(z)
#             pred = self.model.forward(gen_inst)
#             scores = self.criterion(pred,z,gen_inst)
#
#         j = np.argmax(scores,axis=-1)
#         best_z = z[j:j+1] #keeps dimesion
#         if self.verbose:
#             print('best z:', j, ' | score: ', scores[j])
#
#
#         # find colsest instance to z in U to label
#         with torch.no_grad():
#             new_x = self.generator(best_z)
#             pred = self.model.forward(new_x)
#
#         if self.verbose:
#             plt.imshow(new_x.detach().numpy()[0][0],cmap='gray')
#             plt.title("Pred {}".format(F.softmax(pred,dim=1)))
#             plt.axis("off")
#             plt.show()
#
#         # select closest instance in unlabeled pool of generated instance by some measure
#         # TODO: distance measure soft coded
#         # l2 distance in x space
#         l2 = torch.norm(self.U_x.view(self.U_x.size(0),-1)-new_x.view(new_x.size(0),-1),
#                         p=2,dim=-1)
#         i = np.argmin(l2)
#
#
#         if self.verbose:
#             # plot best found new instance
#             with torch.no_grad():
#                 pred = F.softmax(self.model(self.U_x[i:i+1]), dim=1)
#             #print("Closest Instance {} Distance {}".format(i, l2[i]))
#             plt.imshow(self.U_x.detach().numpy()[i][0],cmap='gray')
#             plt.title("Closest Instance index: {} | Distance: {} | Pred: {}".format(i, l2[i], pred))
#             plt.axis("off")
#             plt.show()
#         return i
#
#     def optimize(self,z,lr = 0.05, iterations = 250, early_stop_thresh=None):
#         if self.verbose:
#             with torch.no_grad():
#                 gen_inst = self.generator.forward(z)
#                 pred = self.model(gen_inst)
#                 print("initial loss:",self.criterion(pred,z,gen_inst).sum().item())
#         for i in range(1,iterations+1):
#             self.model.zero_grad()
#             self.model.eval()
#             self.generator.zero_grad()
#             self.generator.eval()
#             z.requires_grad = True
#
#             #print("Grad at beginning: ",z.grad) # change to
#             assert(z.grad == None)
#             gen_inst = self.generator.forward(z)
#             pred = self.model.forward(gen_inst)
#
#             loss = self.criterion(pred,z.reshape(z.size(0),-1),gen_inst).sum()
#
#             # stop early if loss smaller than early_stop_thresh
#             if early_stop_thresh is not None and np.abs(loss.item()) < early_stop_thresh:
#                 print("Stopping early at Iteration: {} | Loss: {:.7f} ".format(i,loss.item()))
#                 break
#
#             loss.backward()
#
#             # add gradient of the input * learning_rate
#             with torch.no_grad():
#                 # to make sure that the SGD step is not in the graph for
#                 # further backprop, there might be a more elegant way to do it
#                 z = z + z.grad*lr
#
#             # print progress and plot instance of once a while
#             if self.verbose and (i % 5 == 0 or i == 1 or i == iterations):
#                 print("Iteration: {} | Loss: {:.7f} ".format(i,loss.item()))
#                 print("image plotting not yet working for multiple z's")
#                 # plt.imshow(gen_inst.detach().numpy()[0][0],cmap='gray')
#
#                 # # Hardcoded output -> prediced cass
#                 # # TODO: make it nice
#                 # pred_class =  pred.argmax()
#                 # plt.title("Predicted Class: {} | Score: {}".format(pred_class,pred))
#                 # plt.show()
#
#         if self.verbose:
#             print("--------- Optimimization finished ----------")
#
#         return z
