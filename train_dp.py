import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
'''
changes from train_single.py

1. wrapping net [nn.DataParallel]
2. .to(device) -> .cuda()

'''

'''
also one can restrict GPUs by

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
'''


def create_data_loader_cifar10(batch_size, num_workers):
    transform = transforms.Compose([transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR10(root='/home/mskang/DDPM/torch_ddp/data', train=True,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='/home/mskang/DDPM/torch_ddp/data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                           shuffle=False, num_workers=num_workers)
    return trainloader, testloader


def train(net, trainloader, epochs):
    print("Start training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    num_of_batches = len(trainloader)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            images, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print(f'[Epoch {epoch + 1}/{epochs}] loss: {running_loss / num_of_batches:.3f}')

    print('Finished Training')


if __name__ == '__main__':
    # time : 80.30
    # 4096*2 succeed! but memory [11579 / 9215] unbalanced
    # set output GPU bigger than others
    # but error whe I set output=1 or 'cuda:1' I don't know why......
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    output_device = None
    batch_size = 4096*2
    num_workers = 20
    epochs = 10
    start = time.time()  
    PATH = './single_net.pth'
    trainloader, testloader = create_data_loader_cifar10(batch_size, num_workers)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = torchvision.models.resnet50(None).cuda()
    net = nn.DataParallel(net, output_device = output_device)
    start_train = time.time()
    train(net, trainloader, epochs)
    end_train = time.time()
    # save
    torch.save(net.state_dict(), PATH)

    end = time.time()
    seconds = (end - start)
    seconds_train = (end_train - start_train)
    print(f"Total elapsed time: {seconds:.2f} seconds, \
        Train {epochs} epoch {seconds_train:.2f} seconds")
