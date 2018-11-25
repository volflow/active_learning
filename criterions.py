import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

class Entropy(nn.Module):
    """calculates row wise Entropy score of log_softmax output of a classifier

    Calculates: sum_i (exp(x_i)*x_i) for every row along dim=0
    """
    def __init__(self,verbose=False,eps=1e-18):
        super(Entropy, self).__init__()
        self.verbose = verbose
        self.eps = eps
    def forward(self, x, *args):
        """calculates entropy of x when x
        """
        x = x# + self.eps

        b = torch.exp(x)*x # x is log softmax
        b = -1.0 * b.sum(dim=-1)

        if self.verbose:
            print("Entropy:",b)
        return b# + self.eps

class Uncertainty(nn.Module):
    """calculates row wise uncertainty score of tensor

    Calculates: -max_x_i(x) for every row along dim=0
    """
    def __init__(self,verbose=False):
        super(UncertaintyLoss, self).__init__()
        self.verbose = verbose
    def forward(self,x,*args):
        x = F.softmax(x,dim=1) # turn x to prob vector
        u,_ = torch.max(x,dim=1)
        u = -1.*u
        if self.verbose:
            print("Uncertainty:",u)
        return u


class Margin(nn.Module):
    """Margin score:
    Calculates (largest x_i) - (second largest x_j) for every row along dim=0
    """
    def __init__(self,verbose=False):
        super(MarginLoss, self).__init__()
        self.verbose = verbose
    def forward(self,x,*args):
        x = F.softmax(x,dim=1) # turn x to prob vector
        top2,_ = torch.topk(x,2,dim=1)
        margin = top2[:,1]-top2[:,0]
        if self.verbose:
            print("Margin:",margin)
            print("Shape:",margin.shape)
        return margin


class MultivariateLogProb(nn.Module):
    """Multivariate Logorithmic Probrobabiliy

    Calculates log probability that z was generated from a multivariate gaussian
    with mean torch.zeros(nz) and cov torch.eye(nz) for every row along dim=0
    """
    def __init__(self,nz=100):
        super(MultivariateLogProb, self).__init__()
        mean=torch.zeros(nz)
        cov=torch.eye(nz)
        self.dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)

    def forward(self, z, *args):
        z = torch.tensor(z,dtype=torch.float)
        return self.dist.log_prob(z)

class DensityWeightedUncertainty(nn.Module):
    """ denisty * uncertainty

    Calculates log(H(pred) * p(z)) = log(H(pred)) + log(p(z))
    with p ~ N(mean,cov)
    """
    def __init__(self,uncertainty=Entropy(),density=MultivariateLogProb(100),verbose=False):
        super(DensityWeightedUncertainty, self).__init__()
        self.uncertainty = uncertainty
        self.density = density

        self.verbose=verbose

    def forward(self, pred, z, x=None):
        u = torch.log(self.uncertainty(pred)+1e-18) #self.uncertainty(pred)
        u -= torch.min(u)
        u /= torch.max(u)+1e-18

        d = self.density(z.reshape(z.size(0),-1))
        d -= torch.min(d)
        d /= torch.max(d)+1e-18
        if self.verbose:
            print("u:",u.shape,' d:',d.shape)
            print(u)
            print(d)
        return u + d

class DiscriminatorWeigthedUncertainty(nn.Module):
    """
    calculates uncertainty(pred)*D(x)
    """
    def __init__(self, discriminator,verbose=False):
        super(DiscriminatorWeigthedUncertainty, self).__init__()
        self.uncertainty = Entropy() #lambda x: torch.log(Entropy()(x))
        self.discriminator = discriminator
        self.verbose = verbose

    def forward(self, pred, z, x):
        #print(pred.shape, z.shape)
        self.discriminator.zero_grad()
        self.discriminator.eval()
        u = self.uncertainty(pred).view(-1)
        r = self.discriminator(x).view(-1)
        score = u * r

        if self.verbose:
            print(x.shape)
            print("Score {} | U: {:.4f} | R: {:.6f}".format(score,u,r.item()))

        return score

class Diversity(nn.Module):
    def __init__(self, normalize=False):
        super(Diversity,self).__init__()
        self.normalize = normalize

    def forward(self,pred,U_z,L_z,lambda_=None):
        """
        TODO: make forward(self,U_z,L_z) work with an Model Agnostic Active Learner
        """

        if len(L_z) > 0:
            dists = self.calc_distances(U_z,L_z)
            if self.normalize:
                dists -= torch.min(dists)
                dists /= torch.max(dists)+1e-18
        else: # special case for picking first instance
            dists = torch.ones(len(U_z))

        return dists

    def calc_distances(self,U_z,L_z):
        """
        calcluate minimal l2 distance of each z in U_z to a z' in L_z
        for each z in U_x
            min_{z' \in L_z} ||z-z'||_2
        """
        l2 = torch.ones(len(U_z))
        for i in range(len(U_z)):
            l2[i] = torch.norm(L_z.view(L_z.size(0),-1)-U_z[i].view(1,U_z[i].size(0)),p=2,dim=-1).min()
        return l2

class DiversityCached(nn.Module):
    """cached diverstiy implementation for efficency

    update_cache(i) needs to be called every time a instance x gets labeled,
        where i is the index of x in U_z
    """
    def __init__(self, U_z, L_z, normalize=False):
        super(DiversityCached,self).__init__()
        self.normalize = normalize
        self.U_z = U_z.view(U_z.size(0),-1)
        self.dists_cache = None
        if L_z is not None and len(L_z) > 0:
            # catch edgecase where L_z is empty in the beginning
            # initialize dists_cache
            self.dists_cache = self.calc_distances(self.U_z,L_z)

    def update_cache(self,i):
        """
        updates the cached distances
        i: int or iterable of indeices of instances in U_z that were added to labeled set
        """
        if type(i) == int:
            i = [i]
        new_labeled = self.U_z[i]
        not_i = [x for x in range(len(self.U_z)) if x not in i]#list(range(i))+list(range(i+1,len(self.U_z)))
        self.U_z = self.U_z[not_i,:]
        l2_new_labeled = self.calc_distances(self.U_z,new_labeled)#,torch.norm(self.U_z-new_labeled.view(len(i),-1),p=2,dim=-1)
        if self.dists_cache is not None:
            # dists_cache has been initialized
            self.dists_cache = torch.min(self.dists_cache[not_i],l2_new_labeled)
        else:
            # initializing dists_cache
            self.dists_cache = l2_new_labeled

    def forward(self,*_):
        """
        TODO: make forward(self,U_z,L_z) work with an Model Agnostic Active Learner
        """
        if self.dists_cache is not None: # dists_cache has been initialized
            dists = self.dists_cache
            if self.normalize:
                dists -= torch.min(dists)
                dists /= torch.max(dists)+1e-18
            return dists
        else:
            return torch.ones(len(self.U_z))

    def calc_distances(self,U_z,L_z):
        """
        calcluate minimal l2 distance of each z in U_z to all z' in L_z
        for each z in U_x
            min_{z' \in L_z} ||z-z'||_2
        """
        l2 = torch.ones(len(U_z))
        for i in range(len(U_z)):
            l2[i] = torch.norm(L_z.view(L_z.size(0),-1)-U_z[i].view(1,U_z[i].size(0)),p=2,dim=-1).min()
        return l2

class DiversityDensityCached(DiversityCached):
    """cached diverstiy*density implementation for efficency

    WORK IN PROGRESS
    """
    def __init__(self,density,U_z, L_z,normalize=False):
        super(DiversityDensityCached,self).__init__(U_z, L_z,normalize)
        self.density = density

    def forward(self,pred,U_z,*_):
        dens = (self.density(self.U_z.reshape(U_z.size(0),-1)))
        #print('desn:',dens.sum())

        if self.dists_cache is not None:
            # dists_cache has been initialized
            div = self.dists_cache
            div = torch.log(div+1e-18)
        else:
            # initializing dists_cache
            div = torch.zeros(len(self.U_z))

        #print('div:',div)

        dd = torch.exp(dens+div)
        #print('dd:',dd)
        if self.normalize:
            dd -= torch.min(dd)
            dd /= torch.max(dd)+1e-18
        return dd

class DiversityDensity(nn.Module):
    def __init__(self,
                density=MultivariateLogProb(100),
                diversity=Diversity(normalize=False),
                normalize=True):
        super(DiversityDensity,self).__init__()
        self.density = density
        self.diversity = diversity
        self.normalize = normalize

    def forward(self,pred,U_z,L_z,lambda_=None):
        """
        TODO: make forward(self,U_z,L_z) work with an Active Learner
        """
        dens = self.density(U_z.reshape(U_z.size(0),-1))

        div = self.diversity(None,U_z=U_z,L_z=L_z)
        div = torch.log(div+1e-18)

        dd = torch.exp(dens+div)
        #print('dd:',dd)
        if self.normalize:
            dd -= torch.min(dd)
            dd /= torch.max(dd)+1e-18
        return dd

class DiversityUncertainty(nn.Module):
    """Uncertainty(pred) + Diversity(z)"""
    def __init__(self,
                uncertainty=Entropy(),
                density=MultivariateLogProb(100),
                verbose=False):
        super(DiversityUncertainty, self).__init__()
        self.uncertainty = uncertainty
        self.density = density
        self.diversity = Diversity(normalize=True)
        self.verbose = verbose

    def forward(self, pred, U_z, L_z, lambda_=1):
        u = self.uncertainty(pred)#torch.log(self.uncertainty(pred))#
        u -= torch.min(u)
        u /= torch.max(u)+1e-18

        d = self.diversity(pred,U_z, L_z)

        return lambda_*u + d


class DiversityDensityUncertainty(nn.Module):
    """Uncertainty(pred) + (Density(z)*Diversity(z))"""
    def __init__(self,
                uncertainty=Entropy(),
                density=MultivariateLogProb(100),
                verbose=False):
        super(DiversityDensityUncertainty, self).__init__()
        self.uncertainty = uncertainty
        self.density = density
        self.divdens = DiversityDensity(
                        density=density,
                        diversity=Diversity(normalize=False),
                        normalize=True)
        self.verbose = verbose

    def forward(self, pred, U_z, L_z, lambda_=1):
        u = self.uncertainty(pred)#torch.log(self.uncertainty(pred))#
        u -= torch.min(u)
        u /= torch.max(u)+1e-18

        dd = self.divdens(pred,U_z, L_z)

        # print('dd:',dd)
        # print('pred:',pred)
        # print('u sum:',pred.sum(dim=-1))
        # print('u:',u)

        #out = self.lambda_*u + (1-self.lambda_)*(d * dists) #29 -> 96.5
        #out = self.lambda_*u + (d * dists)
        #out = (d * dists)
        #out = (lambda_*u + (dists*d))
        out = (lambda_*u + (dd))
        #print('score:',out)
        return out
