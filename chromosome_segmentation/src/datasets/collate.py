def detection_collate(batch):
    """Keep variable-sized images and targets as lists for torchvision detection."""
    return tuple(zip(*batch))
