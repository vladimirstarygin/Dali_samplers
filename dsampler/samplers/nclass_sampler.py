import os
import torch
import random
import imageio
import numpy as np
from ..utils.annotation import get_annotation, create_bucket
from ..utils.sample_utils import create_class_weights

class NClassRandomSampler(object):
    def __init__(self, image_dir, anno_path, 
                 anno_type='.json', batch_size=32, shuffle = True, need_translation=True, min_class_samples = 4):

        self.shuffle = shuffle        
        self.images_dir = image_dir
        self.batch_size = batch_size
        self.annotation = get_annotation(anno_path,anno_type,need_translation)

        self.bucket = create_bucket(self.annotation)
        self.labels = list(self.bucket.keys())
        self.min_samples = min_class_samples
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
        
        labs = random.sample(self.labels,len(self.labels)) 

        for _ in range(self.batch_size // self.min_samples):
            label = labs[_]
            paths = np.random.choice(self.bucket[label],self.min_samples)
            for path in paths:
                batch.append(np.fromfile(os.path.join(self.images_dir, path), dtype = np.uint8))  
                labels.append(torch.tensor([int(label)], dtype = torch.uint8)) 
                self.i += 1

        if self.batch_size % self.min_samples != 0:
            label = labs[-1]
            paths = np.random.choice(self.bucket[label],self.batch_size % self.min_samples)
            for path in paths:
                batch.append(np.fromfile(os.path.join(self.images_dir, path), dtype = np.uint8))  
                labels.append(torch.tensor([int(label)], dtype = torch.uint8)) 
                self.i += 1


        return (batch, labels)