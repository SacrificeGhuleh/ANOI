import torch
import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# We need a dataset. It is split into training set and testing set.

train_data = ((0.111, 0.935), (0.155, 0.958), (0.151, 0.960), (0.153, 0.955),  # # - square
              (0.715, 0.924), (0.758, 0.964), (0.725, 0.935), (0.707, 0.913),  # * - star
              (0.167, 0.079), (0.215, 0.081), (0.219, 0.075), (0.220, 0.078),)  # ## - rectangle

train_labels = ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),  # # - square
                (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0),  # * - star
                (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0),)  # ## - rectangle

test_data = ((0.115, 0.995), (0.210, 0.080), (0.696, 0.948),
             (0.152, 1.000), (0.120, 0.075), (0.159, 0.073),
             (0.732, 0.934), (0.745, 0.954), (0.135, 0.993))

colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')

class MyDataset(Dataset):
  def __init__(self, features, labels):
    self.labels = labels
    self.features = features

  def __len__(self):
    return len(self.features)

  def __getitem__(self, index):
    X = self.features[index]
    y = self.labels[index]

    return X, y


# TODO Here you have to define a neural network model. The configuration can be
#  the same as in your own implementation of neural network.
#  You mainly use torch.nn module or torch.nn.functional module.
class FeedforwardNeuralNetModel(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(FeedforwardNeuralNetModel, self).__init__()
    self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
    self.relu1 = torch.nn.ReLU()
    self.fc2 = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu1(out)
    out = self.fc2(out)

    return out

  # create forward method


# TODO Write the code for training your network here.
# You definitely need PyTorch Algorithms namely zero_grad(), step() and Autograd's backward()
# https://www.programcreek.com/python/example/107701/torch.nn.MSELoss
def train(dataloader, model):
  model.train()

  lr = 0.1
  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=lr)

  avgError = 1
  iteration = 0
  while avgError > 0.001:
    avgError = 0
    for i, sample in enumerate(dataloader):
      feature = sample[0]
      label = sample[1]

      output = model(feature)
      loss = criterion(output, label)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      avgError += loss.item()

    avgError /= 3.0
    iteration += 1
    print('Iteration: {} Error: {}'.format( iteration, avgError))

# TODO Write the code for validating your network here.
def validation(model):
  model.eval()

  for sample in test_data:
    output = model(torch.Tensor(sample))
    predicted = torch.max(output.data, 0)
    print(sample, ' -> ', predicted[1].item())
    plt.scatter(sample[0], sample[1], c=colors[predicted[1].item()])

  plt.show()


if __name__ == "__main__":
  tensor_x = torch.stack([torch.Tensor(i) for i in train_data])
  tensor_y = torch.stack([torch.Tensor(i) for i in train_labels])

  dataset_train = MyDataset(tensor_x, tensor_y)

  dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)

  model = FeedforwardNeuralNetModel(2, 4, 3)

  train(dataloader_train, model)
  validation(model)

