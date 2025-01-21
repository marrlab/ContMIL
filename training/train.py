import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from DataLoader import Dataloader
from model import AMiL
import torch.optim as optim
import os
import time
import numpy as np

# from utils import loaddata

# data = loaddata()

train_loader = torch.utils.data.DataLoader(Dataloader(train=True), num_workers=1)
test_loader = torch.utils.data.DataLoader(Dataloader(train=False), num_workers=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ngpu = torch.cuda.device_count()
print("Found device: ", ngpu, "x ", device)

model = AMiL()

if (ngpu > 1):
    model = torch.nn.DataParallel(model)
model = model.to(device)
print("Setup complete.")
print("")

##### set up optimizer and scheduler
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, nesterov=True)
scheduler = None
epochs = 50

s_test = 0
# Training
loss_func = nn.CrossEntropyLoss()

for ep in range(epochs):
    model.train()

    corrects = 0
    mil_train_loss = 0
    totloss = 0

    time_pre_epoch = time.time()

    for bag, label in train_loader:
        optimizer.zero_grad()

        # print(bag.shape)
        # send to gpu
        label = label.to(device)
        bag = bag.to(device).squeeze()
        prediction, att_softmax = model(bag)

        loss = nn.NLLLoss()(nn.LogSoftmax(1)(prediction), label)
        totloss += loss.data

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # transforms prediction tensor into index of position with highest value
        label_prediction = torch.argmax(prediction, dim=1).item()
        label_groundtruth = label.item()

        if (label_prediction == label_groundtruth):
            corrects += 1

    samples = len(train_loader)
    mil_train_loss /= samples
    totloss /= samples

    accuracy = corrects / samples

    print('- ep: {}/{}, loss: {:.3f}, acc: {:.3f}, {}s'.format(
        ep + 1, epochs, totloss.cpu().detach().numpy(),
        accuracy, int(time.time() - time_pre_epoch)), end=' ')

    # Testing
    model.eval()

    # initialize data structures to store results
    corrects = 0
    train_loss = 0.
    time_pre_epoch = time.time()
    confusion_matrix = np.zeros((9, 9), np.int16)
    # data_obj = DataMatrix()

    optimizer.zero_grad()
    backprop_counter = 0
    loss_func = nn.CrossEntropyLoss()

    for bag, label in test_loader:

        # send to gpu
        label = label.to(device)
        bag = bag.to(device)

        bag = bag.squeeze()

        # forward pass
        prediction, att_softmax = model(bag)

        loss_out = loss_func(prediction, label)
        train_loss += loss_out.data

        # transforms prediction tensor into index of position with highest value
        label_prediction = torch.argmax(prediction, dim=1).item()
        label_groundtruth = label.item()

        if (label_prediction == label_groundtruth):
            corrects += 1
            confusion_matrix[label_groundtruth, label_prediction] += int(1)

            # print('----- loss: {:.3f}, gt: {} , pred: {}, prob: {}'.format(loss_out, label_groundtruth, label_prediction, prediction.detach().cpu().numpy()))

    samples = len(test_loader)
    train_loss /= samples

    accuracy = corrects / samples

    print('test_loss: {:.3f}, test_acc: {:.3f}, {}s'.format(
        train_loss.cpu().numpy(),
        accuracy, int(time.time() - time_pre_epoch)))

torch.save(model, os.path.join("models", "milmodel.pt"))
# print(confusion_matrix)