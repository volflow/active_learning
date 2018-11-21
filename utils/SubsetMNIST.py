from __future__ import print_function
from torchvision.datasets.mnist import MNIST

class SubsetMNIST(MNIST):
    """
    Selects a subset containing only specified classes of the original MNIST dataset
    If re_label=True the new labels will be reassigned from 0 to len(classes)-1
    by classes.index(original_label)
    """
    def __init__(self, root, classes, re_label=False, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)

        self.classes = classes
        select = lambda x: x in self.classes
        new_label = lambda original_label: classes.index(original_label) # call self.classes[x] for original label

        if self.train:
            indices = self.train_labels.clone().apply_(select)
            self.train_data, self.train_labels = self.train_data[indices==1], self.train_labels[indices==1]
            if re_label:
                self.train_labels.apply_(new_label)
        else:
            indices = self.test_labels.clone().apply_(select)
            self.test_data, self.test_labels = self.test_data[indices==1], self.test_labels[indices==1]
            if re_label:
                self.test_labels.apply_(new_label)
