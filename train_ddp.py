import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

## belows are added
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

'''
changes from train_single.py
1. init_disributed()
2. wrapping net [nn.parallel.DistributedDataParallel]
3. .to(device) -> .cuda()
4. dataloader sampler
5. add [trainloader.sampler.set_epoch(epoch)] at the beginning of each epoch
'''

'''
also one can restrict GPUs by

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
'''

'''
dp : same PID
ddp : different PIDs
'''

def init_distributed():
    
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()


def create_data_loader_cifar10(batch_size, num_workers):
    transform = transforms.Compose([transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR10(root='/home/mskang/DDPM/torch_ddp/data', train=True,
                                           download=True, transform=transform)
    train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=False, num_workers=num_workers, pin_memory=True,
                                           sampler = train_sampler)

    testset = torchvision.datasets.CIFAR10(root='/home/mskang/DDPM/torch_ddp/data', train=False,
                                       download=True, transform=transform)
    test_sampler = DistributedSampler(dataset=testset, shuffle=True)                                  
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                           shuffle=False, num_workers=num_workers,
                                           sampler = test_sampler)
    return trainloader, testloader


def train(net, trainloader, epochs):
    print("Start training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    num_of_batches = len(trainloader)

    for epoch in range(epochs):  # loop over the dataset multiple times
       # NEW line added 
        trainloader.sampler.set_epoch(epoch)


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
    # 4096*2 failed! 
    # set output GPU bigger than others
    # but error whe I set output=1 or 'cuda:1' I don't know why......
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

    init_distributed()

    output_device = None
    batch_size = 4096*2
    num_workers = 20
    epochs = 10
    start = time.time()  
    PATH = './single_net.pth'
    trainloader, testloader = create_data_loader_cifar10(batch_size, num_workers)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    net = torchvision.models.resnet50(None).cuda()
    local_rank = int(os.environ['LOCAL_RANK'])
    net = nn.DataParallel(net, device_ids=[local_rank], output_device = output_device)

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
