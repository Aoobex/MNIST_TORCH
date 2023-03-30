# name:Aoobex

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

# use gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get_data

train_data = torchvision.datasets.MNIST("./data", train=True, transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data = torchvision.datasets.MNIST("./data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

# load_data
train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)


# define module
class aoobex_module(nn.Module):
    def __init__(self):
        super(aoobex_module, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
        )

    def forward(self, input):
        return self.model(input)


# create module
module = aoobex_module()
module = module.to(device)

# defined loss function
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)

# define optim
optim = torch.optim.SGD(module.parameters(), lr=0.01)

# define epoch(train times)
epoch = 10

# begin learn
learn_times = 0
train_times = 0
for i in range(epoch):
    module.train()
    print("----------第{}轮训练----------".format(i + 1))
    for data in train_data_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = module(imgs)
        # get loss
        loss = loss_function(output, targets)

        # optim
        optim.zero_grad()
        loss.backward()
        optim.step()

        train_times += 1
        if train_times % 100 == 0:
            print("训练次数:{},损失值:{}".format(train_times, loss))
    print("----------第{}轮训练完毕----------".format(i + 1))
    learn_times += 1
    # save train
    torch.save(module, "train_{}".format(learn_times))
    # train finish,test begin
    module.eval()
    total_loss = 0
    accuracy = 0
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = module(imgs)
            loss = loss_function(outputs, targets)
            total_loss += loss
            accuracy += (outputs.argmax(1) == targets).sum()
    print("总体损失度：{}，正确率：{}".format(total_loss, accuracy / len(test_data)))
