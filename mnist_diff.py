from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

class DiffSuMNISTDataset(Dataset):
    '''Combines two sets of images with labels into a single
    dataset with two images as inputs and a combined label.'''
    
    def __init__(self, x1_labels, x1_images, x2_labels, x2_images):
        
        self.x1 = x1_images
        self.x2 = x2_images
        
        # Create a list of binary values the same length as the dataset
        self.switch_var = torch.randint(0, 2, (len(x1_labels),))
        
        # Create labels
        # If switch_var is 0, add the two labels
        # If switch_var is 1, subtract the second label from the first
        # Add 9 so that category is positive
        self.labels = [(d-e)+9 if s else d+e+9 for s,d,e in zip(self.switch_var, x1_labels, x2_labels)]
    
    def __len__(self):
        return(len(self.labels))
            
    def __getitem__(self, idx):
        x1 = self.x1[idx]
        x2 = self.x2[idx]
        x3 = self.switch_var[idx]
        label = self.labels[idx]
        
        return x1, x2, x3, label

class Net(nn.Module):
    '''Defines the structure of the neural network.'''
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(257, 128)
        self.fc3 = nn.Linear(128, 28) # Output will be between -9 (0-9) and +18 (9+9)

    def forward(self, x1, x2, x3): # x3 is our new binary variable
        x1 = self.conv(x1) 
        x2 = self.conv(x2)
        y = torch.cat((x1, x2), 1)
        y = y.view(-1, 256)
        y = torch.cat((y, x3), 1)
        y = self.fc2(y)
        y = F.relu(y)
        y = self.fc3(y)
        return y
    
    def conv(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x   
    
def gen_combined_dataset(images, labels):
    '''Split a dataset into two datasets, one derived from all even indices and
    one derived from all odd indices.'''
    
    # Dataset 1 - even indices
    x1_labels = labels[range(0, len(labels), 2)]
    x1_images = images[range(0, len(images), 2)]

    # Dataset 2 - odd indices
    x2_labels = labels[range(1, len(labels), 2)]
    x2_images = images[range(1, len(images), 2)]
    
    dataset = DiffSuMNISTDataset(x1_labels, x1_images, x2_labels, x2_images)
    
    return dataset

def plot_cm(y_true, y_pred):
    '''Plots a normalised confusion matrix.'''
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalise
    cm = cm / cm.astype('float').sum(axis=1)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix for DiffSuMNist task")
    plt.colorbar()
    ticks = range(0, 28)
    plt.xticks(ticks, ticks, rotation=45)
    plt.yticks(ticks, ticks)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')

def test(model, device, test_loader, epoch, test_type='Validation', make_cm=False):
    '''Runs inference over a test set. Can be used for validation, or
    for final testing.'''
    
    model.eval()
        
    total_loss = 0
    total_acc = 0
    
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        
        # Iterate over each batch
        for batch in test_loader:

            # Get images and labels
            x1, x2, x3, y = batch

            # Reshape into num_samples, channels, height, width
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)

            x1, x2, x3, y = x1.to(device, dtype=torch.float32), x2.to(device, dtype=torch.float32), x3.to(device), y.to(device)
            
            x3 = x3.reshape(-1, 1)
            
            outputs = model(x1, x2, x3)
            
            values, indices = torch.max(outputs.data, 1)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, y)

            total_loss += loss
            total_acc += torch.sum(indices == y)
            
            all_true = np.concatenate((all_true, y.cpu()), 0)
            all_pred = np.concatenate((all_pred, indices.cpu()), 0)

    print("{} loss for epoch {}: {}".format(test_type, epoch+1, total_loss/len(test_loader.dataset)))
    print("{} accuracy for epoch {}: {}".format(test_type, epoch+1, total_acc/len(test_loader.dataset)))
    
    if(make_cm):
        plot_cm(all_true, all_pred)

def train(model, device, train_loader, val_loader, optimizer, num_epochs):
    '''Trains the model, producing validation metrics after each epoch.'''
    
    model.train()
    
    for epoch in range(num_epochs):
        
        print('Epoch {} of {}'.format(epoch+1, num_epochs))
        
        epoch_loss = 0
        epoch_acc = 0
        
        # Iterate over each batch
        for batch in train_loader:
            
            # Get images and labels
            x1, x2, x3, y = batch
            
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)
            
            x1, x2, x3, y = x1.to(device, dtype=torch.float32), x2.to(device, dtype=torch.float32), x3.to(device), y.to(device)
            
            x3 = x3.reshape(-1, 1)
            
            optimizer.zero_grad()

            outputs = model(x1, x2, x3)
            
            # Output will be a vector with length 28
            # Get index of max value
            values, indices = torch.max(outputs.data, 1)
            
            # CrossEntropyLoss calculates the distances between the activations and the target
            # Target can be an index i.e. if target is 15 assume output should be 0 for all other indices
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, y)

            epoch_loss += loss
            epoch_acc += torch.sum(indices == y)
            
            loss.backward()
            optimizer.step()
            
        print("Loss for epoch {}: {}".format(epoch+1, epoch_loss/len(train_loader.dataset)))
        print("Accuracy for epoch {}: {}".format(epoch+1, epoch_acc/len(train_loader.dataset)))
        
        test(model, device, val_loader, epoch)
        
    return model

def main():
    
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch SuMNIST Project')
    parser.add_argument('--mode', type=str, default='train', metavar='M',
                        choices=['train', 'test'], help='mode (train or test) (default: train)')
    parser.add_argument('--cm', action='store_true', default=False,
                        help='whether or not to output a confusion matrix')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if use_cuda else "cpu")
    
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # Normalise using pre-computed mean and STD
            ])
    
    if(args.mode == 'train'):
    
        train_kwargs = {'batch_size': args.batch_size}
        val_kwargs = {'batch_size': args.val_batch_size}
    
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            val_kwargs.update(cuda_kwargs)
        
        # Fetch training set
        all_train = datasets.MNIST('../mnist_data', train=True, download=True, transform=transform)
    
        # Subset training and validation sets
        # 50,000 for training, 10,000 for validation
        train_labels = all_train.targets[:50000]
        train_imgs = all_train.data[:50000]
        val_labels = all_train.targets[50000:]
        val_imgs = all_train.data[50000:]
    
        # Create training and validation data sets.
        # These are created by splitting each set into two, so 
        # that each input image pair is unique
        train_set = gen_combined_dataset(train_imgs, train_labels)
        val_set = gen_combined_dataset(val_imgs, val_labels)
    
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
        val_loader = torch.utils.data.DataLoader(val_set, **train_kwargs)
    
        # Create model
        model = Net().to(device)
    
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
        trained_model = train(model, device, train_loader, val_loader, optimizer, args.epochs)
    
        # Save model for later testing
        if args.save_model:
            torch.save(trained_model.state_dict(), "mnist_cnn.pt")
            
    elif(args.mode == 'test'):
        
        # Get training set
        all_test_data = datasets.MNIST('../mnist_data', train=False, download=True, transform=transform)
        
        test_kwargs = {'batch_size': args.test_batch_size}
    
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            test_kwargs.update(cuda_kwargs)

        # Extract labels and images
        test_labels = all_test_data.targets
        test_imgs = all_test_data.data
        
        # Create new combined dataset
        test_set = gen_combined_dataset(test_imgs, test_labels)
        
        # Create loader
        test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)
        
        # Load trained model
        model = Net().to(device)
        model.load_state_dict(torch.load("mnist_cnn.pt"))
        
        test(model, device, test_loader, 0, test_type='Test', make_cm=args.cm)
        
if __name__ == '__main__':
    main()