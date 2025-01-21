import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from DataLoader import Dataloader
import sklearn.metrics as metrics

dlt = Dataloader(train=False)
test_loader = torch.utils.data.DataLoader(dlt, num_workers=1)

model = torch.load("models/milmodel.pt", map_location="cpu")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ngpu = torch.cuda.device_count()
print("Found device: ", ngpu, "x ", device)

if (ngpu > 1):
    model = torch.nn.DataParallel(model)
model = model.to(device)
print("Setup complete.")
print("")

model.eval()
gt = torch.tensor([]).to(device)
preds = torch.tensor([]).to(device)
for bag, label in test_loader:
    label = label.to(device).unsqueeze(0)
    bag = bag.squeeze().to(device)
    prediction,_ = model(bag)
    preds = torch.cat([preds, torch.argmax(prediction, dim=1)])

    gt = torch.cat([gt, label])
gt = gt.cpu().detach().numpy()
preds = preds.cpu().detach().numpy()
accuracy = metrics.accuracy_score(gt, preds)
conf = metrics.confusion_matrix(gt,preds)
print(accuracy)
print(conf)
