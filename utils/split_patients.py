import pickle

import numpy as np
from sklearn.model_selection import train_test_split

with open("../dataset/patients.dat", "rb") as f:
    patients = pickle.load(f)

labels = {'Acute leukaemia':0,
          'Lymphoma':1,
          'MDS':2,
          'MDS / MPN':3,
          'MPN':4,
          'No malignancy':5,
          'Plasma cell neoplasm':6
          }

for p in list(patients):
    if patients[p]["label_conv"] not in list(labels.keys()):
        patients.pop(p)

patient_names = list(patients.keys())
patient_labels = [patients[p]["label"] for p in patients]

trainlist, testlist, _, testlbls = train_test_split(patient_names, patient_labels,
                                             stratify=patient_labels,
                                             test_size=0.25,
                                             random_state=42)

with open("../dataset/split.dat", "wb") as f:
    pickle.dump([trainlist, testlist], f)
print(len(trainlist), len(testlist))
print("done")

print(np.unique(testlbls, return_counts=True))