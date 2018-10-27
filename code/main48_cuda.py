from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device: ", device)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--wd', type=float, default=0, metavar='N',
                    help='weight decay used in the model')
parser.add_argument('--save_to', type=str, default='./', metavar='N',
                    help='path to save models')
parser.add_argument('--load_from', type=str, default='', metavar='N',
                    help='path to the pth of a file, to continue training it')
args = parser.parse_args()
torch.manual_seed(args.seed)

### Data Initialization and Loading
from data48 import initialize_data, data_transforms # data48.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=0)

### Neural Network and Optimizer
# We define neural net in model48.py so that it can be reused by the evaluate script
from model48 import Net
model = Net().to(device)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

first_epoch = 1
if args.load_from != '':
    first_epoch = int(filter(str.isdigit, args.load_from))
    model.load_state_dict(torch.load(args.load_from))

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-6, nesterov=True)

def train(epoch):
    avg_loss = 0
    steps = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        steps += 1
        avg_loss += loss.data[0].detach().cpu().numpy()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

	#if epoch % 10 == 0:
	lr = args.lr * (args.wd ** (epoch / 10))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

    return float(avg_loss) / steps

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True).to(device), Variable(target).to(device)
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    return validation_loss.detach().cpu().numpy()

losses = []
for epoch in range(first_epoch, args.epochs + 1):
    train_loss = train(epoch)
    val_loss = validation()
    losses.append((epoch, train_loss, val_loss))
    model_file = args.save_to + 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    pickle.dump(losses, open(args.save_to + 'losses.p', 'wb'))
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')
	

