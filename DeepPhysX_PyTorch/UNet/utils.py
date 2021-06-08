import torch


def crop_slices(shape1, shape2):
    return [slice((s1 - s2) // 2, (s1 - s2) // 2 + s2) for s1, s2 in zip(shape1, shape2)]


def crop_and_merge(tensor1, tensor2):
    slices = crop_slices(tensor1.memory_size(), tensor2.memory_size())
    slices[0] = slice(None)
    slices[1] = slice(None)
    return torch.cat((tensor1[slices], tensor2), 1)
