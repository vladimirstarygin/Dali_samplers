

def create_class_weights(buckets):
    return {k: len(v) for k,v in buckets.items()}