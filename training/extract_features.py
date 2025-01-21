import pickle
import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision.models
from tqdm import tqdm
from torchvision.models.resnet import resnet18, ResNet18_Weights

from PatientLoader import PatientLoader

#root = '/lustre/groups/labs/marr/qscd01/datasets/210714_mll_march/March_2021'
root = '/lustre/groups/labs/marr/qscd01/workspace/ario.sadafi/gCont_MIL/BelugaMLL/patients/'
import ctypes

libgcc_s = ctypes.CDLL('libgcc_s.so.1')

cuda = torch.cuda.is_available()

# model = torch.load(
#     "/lustre/groups/aih/oleksandra.adonkina/RestNet_Testing/MIL_Peter/Final/mll_full_res_34_sgd_lr_0005_f2_bs_512_final/model.pt",
#     map_location=torch.device('cpu'))


model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

if cuda:
    model = model.cuda()
model.eval()


def forward(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    return x


loader_kwargs = {'num_workers': 50, 'pin_memory': True} if cuda else {}

features = {}

dataloader = data_utils.DataLoader(PatientLoader(root),
                                   batch_size=1,
                                   shuffle=False,
                                   **loader_kwargs)
pbar = tqdm(dataloader)
for p, imgs in pbar:
    pbar.set_description(p[0])
    imgs = imgs.squeeze().float()
    if len(imgs.shape) < 4:
        print(p[0])
        continue

    if cuda:
        imgs = imgs.cuda()
    feats = forward(model, imgs)
        #print(feats.shape)
#    try:
 #   features[p[0]] = feats.cpu().detach().numpy()
#    except:
#        print(p[0])
#        continue

#     with open("/lustre/groups/labs/marr/qscd01/workspace/ario.sadafi/gCont_MIL/BelugaMLLFeats/mll/"+p[0] +".pkl", "wb") as f:
#         pickle.dump(feats.cpu().detach().numpy(), f)
    np.save("/lustre/groups/labs/marr/qscd01/workspace/ario.sadafi/gCont_MIL/BelugaMLLFeats/resnet18/"+p[0] +".pkl", feats.cpu().detach().numpy())
#np.save("/lustre/groups/labs/marr/qscd01/workspace/ario.sadafi/gCont_MIL/BelugaMLLFeats/mll/feats.pkl", features)
