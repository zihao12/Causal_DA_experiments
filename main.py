import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms

# from colored_mnist import ColoredMNIST
from data_prepare import *

import argparse
import pickle
import pdb
import numpy as np

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(3 * 28 * 28, 512)
    self.fc2 = nn.Linear(512, 512)
    self.fc3 = nn.Linear(512, 1)

  def forward(self, x):
    x = x.view(-1, 3 * 28 * 28)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    logits = self.fc3(x).flatten()
    return logits


class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 20, 5, 1)
    self.conv2 = nn.Conv2d(20, 50, 5, 1)
    self.fc1 = nn.Linear(4 * 4 * 50, 500)
    self.fc2 = nn.Linear(500, 1)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 4 * 4 * 50)
    x = F.relu(self.fc1(x))
    logits = self.fc2(x).flatten()
    return logits


def test_model(model, device, test_loader, set_name="test set"):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device).float()
      output = model(data)
      test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()  # sum up batch loss
      pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
                         torch.Tensor([1.0]).to(device),
                         torch.Tensor([0.0]).to(device))  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  #pdb.set_trace()
  test_loss /= len(test_loader.dataset)

  print('\nPerformance on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    set_name, test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

  return 100. * correct / len(test_loader.dataset)

def eval_model(model, device, test_loader, set_name="test set"):
  model.eval()
  acc = np.array([])
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device).float()
      output = model(data)
      pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
                         torch.Tensor([1.0]).to(device),
                         torch.Tensor([0.0]).to(device))  # get the index of the max log-probability
      acc = np.append(acc, pred.eq(target.view_as(pred)).cpu().detach().numpy())

  print(f'acc on {set_name} is {acc.mean()}')
  return acc


def compute_irm_penalty(losses, dummy, weights):
  if weights is None:
    g = grad(losses.mean(), dummy, create_graph=True)[0]
  else:
    g = grad((weights * losses).mean(), dummy, create_graph=True)[0]
  return (g ** 2).sum()

def irm_train(model, device, train_loaders, optimizer, epoch, penalty_weight, penalty_anneal_iters, generalized):
  model.train()

  #pdb.set_trace()
  train_loaders = [iter(x) for x in train_loaders]
  n_d = len(train_loaders)
  penalties_girm = [0 for i in range(n_d)]
  penalties_irm = [0 for i in range(n_d)]

  dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(device)

  batch_idx = 0
  while True:
    optimizer.zero_grad()
    error = 0
    penalty = 0
    priors = [0.65, 0.35] ## for domain 1, P(Y = 0) = 0.8 

    for d, loader in enumerate(train_loaders):
      data, target = next(loader, (None, None))
      if data is None:
        return penalties_irm, penalties_girm

      prior = priors[d]
      weights = torch.Tensor([0.5 / prior] * len(target))
      weights[target == 1] = 0.5 / (1 - prior)      

      data, target, weights = data.to(device), target.to(device).float(), weights.to(device)
      output = model(data)
      loss_erm = F.binary_cross_entropy_with_logits(output * dummy_w, target, reduction='none')

      penalty_girm = compute_irm_penalty(loss_erm, dummy_w, weights)
      penalty_irm = compute_irm_penalty(loss_erm, dummy_w, None)
      if generalized:
        penalty += penalty_girm
      else:
        penalty += penalty_irm

      penalties_girm[d] += penalty_girm.item()
      penalties_irm[d] += penalty_irm.item()      

      error += loss_erm.mean()

    penalty_weight = (penalty_weight if epoch >= penalty_anneal_iters else 1.0) 
    print(f"penalty weight {penalty_weight}")
    
    (error + penalty_weight * penalty).backward()
    optimizer.step()
    if batch_idx % 2 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tERM loss: {:.6f}\tGrad penalty: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loaders[0]._dataset),
               100. * batch_idx / len(train_loaders[0]), error.item(), penalty.item()))
      print('First 20 logits', output.data.cpu().numpy()[:20])

    batch_idx += 1


def train_and_test_irm(maxiter, out_result_name, out_model_name, penalty_weight, penalty_anneal_iters, generalized):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

  data_train = ColoredMNIST(root='./data', env='all_train',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ]))
  complete_data_loader = torch.utils.data.DataLoader(data_train,
    batch_size=2000, shuffle=False, **kwargs)

  data_train1 = ColoredMNIST(root='./data', env='train1',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ]))

  train1_loader = torch.utils.data.DataLoader(data_train1,
    batch_size=2000, shuffle=True, **kwargs)

  data_train2 = ColoredMNIST(root='./data', env='train2',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ]))

  train2_loader = torch.utils.data.DataLoader(data_train2,
    batch_size=2000, shuffle=True, **kwargs)


  test_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='test', transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
    ])),
    batch_size=1000, shuffle=False, **kwargs)

  model = ConvNet().to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  train1_accs = []
  train2_accs = []
  test_accs = []
  penalties_irm = []
  penalties_girm = []

  for epoch in range(1, maxiter + 1):
    penalties_irm_tmp, penalties_girm_tmp = irm_train(model, device, [train1_loader, train2_loader], optimizer, epoch, penalty_weight, penalty_anneal_iters, generalized)    
    train1_acc = test_model(model, device, train1_loader, set_name='train1 set')
    train2_acc = test_model(model, device, train2_loader, set_name='train2 set')
    test_acc = test_model(model, device, test_loader)

    train1_accs.append(train1_acc)
    train2_accs.append(train2_acc)
    test_accs.append(test_acc)
    penalties_irm.append(penalties_irm_tmp)
    penalties_girm.append(penalties_girm_tmp)

    # if test_acc > 70:
    #   print('found acceptable values. stopping training.')
    #   break

  out = {}
  out['train1_accs'] = train1_accs
  out['train2_accs'] = train2_accs
  out['test_accs'] = test_accs
  out['penalties_irm'] = penalties_irm
  out['penalties_girm'] = penalties_girm
  with open(out_result_name, 'wb') as f:
    pickle.dump(out, f)
  torch.save(model, out_model_name)

  
def plot_dataset_digits(dataset):
  fig = plt.figure(figsize=(13, 16))
  columns = 6
  rows = 6
  # ax enables access to manipulate each of subplots
  ax = []

  for i in range(columns * rows):
    img, label = dataset[i]
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i + 1))
    ax[-1].set_title("Label: " + str(label))  # set title
    plt.imshow(img)

  plt.show()  # finally, render the plot


def main(method, maxiter, out_result_name, out_model_name, penalty_weight, penalty_anneal_iters):
  if method == 'irm':
    train_and_test_irm(maxiter, out_result_name, out_model_name, penalty_weight, penalty_anneal_iters, False)
  if method == 'girm':
    train_and_test_irm(maxiter, out_result_name, out_model_name, penalty_weight, penalty_anneal_iters, True)
  if method == 'erm':
    train_and_test_irm(maxiter, out_result_name, out_model_name, 0, 0, False)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--i', type=int, default= 1, help='model index')
  parser.add_argument('--method', type=str, default='irm',  help='method')
  # parser.add_argument('--g', type=bool, default= False, help='whether to use generalized version of IRM')
  parser.add_argument('--maxiter', type=int, default=1,  help='method')
  parser.add_argument('--out_result', type=str,  default="test.pkl", help='out_result')
  parser.add_argument('--out_model', type=str,  default="test.pt", help='out_model')
  parser.add_argument('--penalty_anneal_iters', type=int, default=100)
  parser.add_argument('--penalty_weight', type=float, default=10000.0)
  args = parser.parse_args()

  method = args.method
  # generalized = args.g
  maxiter = args.maxiter
  i = args.i
  out_result_name = args.out_result
  out_model_name = args.out_model
  penalty_anneal_iters = args.penalty_anneal_iters
  penalty_weight = args.penalty_weight

  torch.manual_seed(i)
  main(method, maxiter, out_result_name, out_model_name, penalty_weight, penalty_anneal_iters)
