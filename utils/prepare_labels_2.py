import csv
import pickle

import numpy as np

# prepare labels
labels = []
with open("../dataset/patient_list_Christian_coarse_noMGUS.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader)  # skip header
    for row in reader:
        arrayid = row[0]
        # pid = row[1]
        labels.append(row[1])

label_names, label_freq = np.unique(labels, return_counts=True)

with open("../dataset/labels.dat", "wb") as f:
    pickle.dump(label_names, f)
print("job finished")
