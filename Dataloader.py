import random
import string
import torch


class Dataloader:
    def __init__(self, task, data, train):
        self.task = task
        self.data = data
        self.train = train
        self.patients = task.patients["trainset"] if train else task.patients["testset"]
        self.patientlist = list(self.patients)

    def __len__(self):
        return len(self.patientlist)

    def __getitem__(self, index):
        pid = self.patientlist[index]
        feats, label = self._get(pid)
        return index, feats, label

    def _get(self, pid):

        feats = self.data[pid]
        label = self.patients[pid]["label"]
        # feats = torch.tensor(feats, dtype=torch.float)
        # label = torch.tensor(label, dtype=torch.long)
        return feats, label

    def get_images(self, cls):
        plist = [p for p in self.patientlist if self.patients[p]["label"] == cls]
        ret = []
        for p in plist:
            ret.append(self._get(p)[0])

        return ret

    def append(self, imgs, key):
        for i, img in enumerate(imgs):
            pid = "icarl-" + key + "-" + str(i) + "-" + ''.join(
                random.choice(string.ascii_lowercase) for i in range(7))
            #  self.data[pid] = imgs
            self.data[pid] = img
            self.patients[pid] = {"label": int(key)}
