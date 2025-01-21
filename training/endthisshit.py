import os.path
import pickle
import numpy as np

FEATPATH = "/lustre/groups/labs/marr/qscd01/workspace/ario.sadafi/gCont_MIL/BelugaMLLFeats/mll/"
labels = {'Reactive': 0,
          'MDS': 1,
          'AML': 2,
          'CML': 3,
          'Normalbefund': 4,
          'CMML': 5,
          'CLL': 6,
          'MPN': 7,
          }


#print info

with open("../dataset/split.dat", "rb") as f:
    lists = pickle.load(f)
with open("../dataset/patients.dat", "rb") as f:
    patients = pickle.load(f)


test_lbls = []
for p in lists[1]:
    if os.path.exists(os.path.join(FEATPATH, patients[p]["array_id"] + ".pkl.npy")):
        test_lbls.append(patients[p]["label_conv"])

print(np.unique(test_lbls, return_counts=True))
print("end")
