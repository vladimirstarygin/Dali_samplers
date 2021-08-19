import os
import torch
import random
import imageio
import numpy as np
from ..utils.annotation import get_annotation

class SimpleSampler(object):
    def __init__(self, image_dir, anno_path, anno_type='.json', batch_size=32, shuffle = True, need_translation=True):

        self.shuffle = shuffle        
        self.images_dir = image_dir
        self.batch_size = batch_size
        self.annotation = get_annotation(anno_path,anno_type,need_translation)

        self.len = len(self.annotation)
    
    def __iter__(self):
        self.i = 0
        if self.shuffle:
            random.shuffle(self.annotation)
        return self

    def __len__(self):
        return self.len

    def __next__(self):

        batch = []
        labels = []

        if self.i >= self.len:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            path, label = self.annotation[self.i % self.len]
            batch.append(np.fromfile(os.path.join(self.images_dir, path), dtype = np.uint8))  
            labels.append(torch.tensor([int(label)], dtype = torch.uint8)) 
            self.i += 1

        return (batch, labels)