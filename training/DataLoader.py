import cv2
import numpy as np
import random
import string
from torch.utils.data import Dataset
import pickle
import os
import numpy as np
from tqdm import tqdm


FEATPATH = "/lustre/groups/labs/marr/qscd01/workspace/ario.sadafi/gCont_MIL/BelugaMLLFeats/resnet18/"
labels = {'Acute leukaemia':0,
          #'Lymphoma':1,
          'MDS':1,
          # 'MDS / MPN':3,
          'MPN':2,
          'No malignancy':3,
          'Plasma cell neoplasm':4
          }


class Dataloader(Dataset):

    def __init__(self, train, use_cache=True):

        with open("../dataset/split.dat", "rb") as f:
            lists = pickle.load(f)
        with open("../dataset/patients.dat", "rb") as f:
            patients = pickle.load(f)
        self.patient_list = lists[0] if train else lists[1]

        for p in list(patients):
            if patients[p]["label_conv"] not in list(labels.keys()):
                if p in self.patient_list:
                    self.patient_list.remove(p)

            if not os.path.exists(os.path.join(FEATPATH, patients[p]["array_id"] + ".pkl.npy")):
                if p in self.patient_list:
                    self.patient_list.remove(p)
                    print("not found", p)

        self.patients = patients
        if use_cache:
            self.patient_data = self._load_or_cache(train)

    def _load_or_cache(self, train):
        fname = "train.dat" if train else "test.dat"
        fpath = "../dataset/" + fname

        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                data = pickle.load(f)

        else:
            data = {}
            for index in tqdm(range(len(self.patient_list))):
                pid = self.patient_list[index]
                feats, label = self._get_from_disk(pid)
                data[pid] = [feats, label]
            with open(fpath, "wb") as f:
                pickle.dump(data, f)
        print(fname, "successfully loaded.")
        l = [data[pid][1] for pid in self.patient_list]
        print(np.unique(l,return_counts=True))
        self.cache_loaded = True
        return data



    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, index):
        pid = self.patient_list[index]
        if self.cache_loaded:
            feats, label = self.patient_data[pid]
        else:
            feats, label = self._get_from_disk(pid)

        return feats, label

    def _get_from_disk(self, pid):
        # feats = self.features[pid]
        # with open(os.path.join(FEATPATH, self.patients[pid]["array_id"] + ".pkl"), "rb") as f:
        #     feats = pickle.load(f)

        feats = np.load(os.path.join(FEATPATH, self.patients[pid]["array_id"] + ".pkl.npy"))

        try:
            label = labels[self.patients[pid]["label_conv"]]
        except:
            print(self.patients[pid])

        return feats, label
