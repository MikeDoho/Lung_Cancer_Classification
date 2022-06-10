import numpy as np

import torch
from torch.utils.data import TensorDataset as dset

torch.manual_seed(42)

data_size = 15
num_classes = 3

batch_size = 4

inputs = torch.tensor(range(data_size))
print("inputs", inputs.shape, inputs)
if 0:
    targets = torch.floor(num_classes * torch.rand(data_size)).int()
else:
    targets = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1, 1, 2, 2, 1, 0, 0, 1], dtype=torch.int32)

print("targets", targets.shape, targets)
trainDataset = dset(inputs, targets)

# https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/10
class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
print("class_sample_count", class_sample_count.shape, class_sample_count)

weights = 1. / class_sample_count
print("weights", weights.shape, weights)

# https://discuss.pytorch.org/t/some-problems-with-weightedrandomsampler/23242/2
samples_weights = weights[targets]
print("samples_weights", samples_weights.shape, samples_weights)
assert len(samples_weights) == len(targets)

if 0:
    print("samples_weights", samples_weights.shape, samples_weights)

sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=batch_size, sampler=sampler)

inputs_new = []
targets_new = []
for batch, (data, target) in enumerate(trainLoader):
    counts = [len(np.where(target.numpy() == class_sample)[0]) for class_sample in range(len(class_sample_count))]
    inputs_new.extend(data.data.numpy())
    targets_new.extend(target.data.numpy())
    print("batch {}, size {}, data {}, counts: {}".format(batch, data.shape[0], target.data, counts))

print("inputs_new", inputs_new)
print("targets_new", targets_new)
print("class_sample_count_new", np.array([len(np.where(targets_new == t)[0]) for t in np.unique(targets_new)]))