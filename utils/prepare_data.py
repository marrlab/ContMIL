import csv
import pickle
import os


PPATH = "/lustre/groups/labs/marr/qscd01/workspace/ario.sadafi/gCont_MIL/BelugaMLL/patients/"


# labels = {'Reactive': 0,
#           'MDS': 1,
#           'AML': 2,
#           'CML': 3,
#           'Normalbefund': 4,
#           'CMML': 5,
#           'CLL': 6,
#           'MPN': 7,
#           }
labels = {'Acute leukaemia':0,
          'Lymphoma':1,
          'MDS':2,
          'MDS / MPN':3,
          'MPN':4,
          'No malignancy':5,
          'Plasma cell neoplasm':6
          }

# with open("../dataset/labels.dat", "rb") as f:
#     labels = pickle.load(f)

patients = {}
# p_arr_pid = {}
#
# with open("../dataset/master_df.csv", "r") as csvfile:
#     reader = csv.reader(csvfile, delimiter=",")
#     next(reader)  # skip header
#     for row in reader:
#         p_arr_pid[row[0]] = row[1]

with open("../dataset/patient_list_Christian_coarse_noMGUS.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader)  # skip header
    for row in reader:
        if os.path.exists(os.path.join(PPATH, row[0])):
            # pid = p_arr_pid[row[0]]
            patients[row[0]] = {
                "array_id": row[0],
                "label_orig": row[1],
                "label_conv": row[1],
                "label": row[1],
                "id": int(row[0].split("_")[1])
            }

with open("../dataset/patients.dat", "wb") as f:
    pickle.dump(patients, f)

print("job finished")
