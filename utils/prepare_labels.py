import csv
import pickle

import numpy as np

# prepare labels
labels = []
with open("../dataset/beluga_all_patients.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader)  # skip header
    for row in reader:
        arrayid = row[0]
        pid = row[1]
        labels.append(row[2])

label_names, label_freq = np.unique(labels, return_counts=True)

label_dict = {}

words = ["AL", "AML", "AMML", "CML", "CMML", "CLL", "MDS", "MPN", "Lymphoma",
         "Reactive", "Reaktive", "Reaktion",
         "Lymphom", "Lymphombefall",  "Lymphoplasmacytic", "Lymphoplasmozytisches", "Lymphoplasmozytisches",
        "Normalbefund",
         # "Normal findings", "No evidence",
         ]
for w in words:
    label_dict[w] = []

sum_lost = 0
for l, ll in zip(label_names, label_freq):
    found = False
    for w in words:
        if w in l:
            label_dict[w].append(l)
            found = True
    if not found:
        print(l, ll)
        sum_lost += ll

label_dict["Lymphoma"].extend(label_dict["Lymphombefall"])
label_dict.pop("Lymphombefall")
label_dict["Lymphoma"].extend(label_dict["Lymphoplasmacytic"])
label_dict.pop("Lymphoplasmacytic")
label_dict["Lymphoma"].extend(label_dict["Lymphoplasmozytisches"])
label_dict.pop("Lymphoplasmozytisches")
label_dict["Lymphoma"].extend(label_dict["Lymphom"])
label_dict.pop("Lymphom")

label_dict["Reactive"].extend(label_dict["Reaktion"])
label_dict.pop("Reaktion")
label_dict["Reactive"].extend(label_dict["Reaktive"])
label_dict.pop("Reaktive")


label_dict.pop("AL")
label_dict.pop("AMML")

print("sum lost", sum_lost)
print("remains:", len(labels) - sum_lost)

label_dict_trans = {}
for key in label_dict:
    for cls in label_dict[key]:
        label_dict_trans[cls] = key

counts = {}
for key in label_dict:
    counts[key] = 0
for lbl in labels:
    if lbl in label_dict_trans.keys():
        counts[label_dict_trans[lbl]] += 1

print(counts)
print(sum(counts.values()))
with open("../dataset/label_dict.dat", "wb") as f:
    pickle.dump([label_dict_trans, label_dict], f)
print("job finished")
