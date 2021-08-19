import json
from tqdm import tqdm 
from collections import defaultdict  

def get_annotation(anno_path, anno_type, need_translation):

    anno_path = [anno_path] if isinstance(anno_path, str) else anno_path

    if anno_type == '.json':
        return json_annotation(anno_path) if not need_translation else translate_dataset(json_annotation(anno_path))

def translate_dataset(annotation):
    labels = set([lab for _, lab in annotation])
    translation_dict = {v: k for k,v in enumerate(labels)}
    return [[path, translation_dict[label]] for path,label in annotation]

def json_annotation(anno_path):
    
    full_annotation = []
    for annotation in anno_path:
        with open(annotation, 'r', encoding = 'utf-8') as f:
            data = json.load(f)
        for path, label in tqdm(data.items(),desc='Reading ' + annotation):
            full_annotation.append([path, label])

    return full_annotation

def create_bucket(annotation):
    bucket = defaultdict(list)
    for path,label in annotation:
        bucket[label].append(path)
    return bucket
    