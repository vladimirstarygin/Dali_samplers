import os
import torch
import random
import imageio
import numpy as np
from ..utils.annotation import get_annotation, create_bucket
from ..utils.sample_utils import create_class_weights

class WeightedRandomSampler(object):
    def __init__(self, image_dir, anno_path, 
                 anno_type='.json', batch_size=32, shuffle = True, need_translation=True):

        self.shuffle = shuffle        
        self.images_dir = image_dir
        self.batch_size = batch_size
        self.annotation = get_annotation(anno_path,anno_type,need_translation)

        self.bucket = create_bucket(self.annotation)
        self.weights_bucket = create_class_weights(self.bucket)
        self.weights = torch.DoubleTensor(list(self.weights_bucket.values()))
        self.weights_class = list(self.weights_bucket.keys())
        print(self.weights_bucket)
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

        idxes = torch.multinomial(self.weights, self.batch_size, replacement=True)

        for idx in idxes:
            label = self.weights_class[idx] 
            path = random.choice(self.bucket[label])
            batch.append(np.fromfile(os.path.join(self.images_dir, path), dtype = np.uint8))  
            labels.append(torch.tensor([int(label)], dtype = torch.uint8)) 
            self.i += 1

        return (batch, labels)