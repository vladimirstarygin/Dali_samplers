import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from collections import Counter
from dsampler.samplers.simple_sampler import SimpleSampler
from dsampler.samplers.random_sampler import RandomSampler
from dsampler.samplers.balanced_sampler import WeightedRandomSampler
from dsampler.samplers.nclass_sampler import NClassRandomOverSampler

from nvidia.dali.plugin.pytorch import DALIClassificationIterator as PyTorchIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy

image_dir_path = ''
annotation_path = ''
sampler = NClassRandomOverSampler(image_dir=image_dir_path, 
              anno_path=annotation_path, 
              anno_type='.json', 
              batch_size=32, 
              shuffle = True)

def ExternalSourcePipeline(batch_size, num_threads, device_id, external_data):
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs, labels = fn.external_source(source=external_data, num_outputs=2)
        images = fn.decoders.image(jpegs, device="mixed")
        images = fn.resize(images, resize_x=240, resize_y=240)
        output = fn.cast(images, dtype=types.UINT8)
        pipe.set_outputs(output, labels)
    return pipe


pipe = ExternalSourcePipeline(batch_size=32, num_threads=2, device_id = 0,
                              external_data = sampler)
pii = PyTorchIterator(pipe, last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL)
epochs = 1
for e in range(epochs):
    for i, data in enumerate(pii):
        labs = data[0]['label'].reshape(-1).numpy()
        c = Counter(labs)
        print(c)
        #print("epoch: {}, iter {}, real batch size: {}".format(e, i, len(data[0]["data"])))
    pii.reset()