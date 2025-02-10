import gzip
import pickle

import numpy as np

from Task import Task


class TaskManager:
    def __init__(self, experiment):
        self.Tasks = {}

        self._split, self.patients = self.load()

        if experiment == 1:
            cls = [[0, 1], [2, 3], [4, 5], [6, 7]]

            cumm_cls = []
            for i, c in enumerate(cls):
                cumm_cls.extend(c)
                self.Tasks[i] = Task(task_id=i,
                                     experiment=experiment,
                                     class_list=c,
                                     split=self._split,
                                     patients=self.patients,
                                     cumm_cls=cumm_cls.copy(),
                                     modelname=str(experiment) + "-" + str(i),
                                     prev_modelname=str(experiment) + "-" + str(i - 1) if i > 0 else "")

        elif experiment == 2:
            cls_list = [0, 1, 2, 3, 4, 5, 6]
            cumm_cls = cls_list
            arrays0 = np.sort(
                [int(self.patients[p]["array_id"].split("_")[1]) for p in self.patients if p in self._split[0]])
            arrays1 = np.sort(
                [int(self.patients[p]["array_id"].split("_")[1]) for p in self.patients if p in self._split[1]])
            weeks = zip(np.array_split(arrays0, 4), np.array_split(arrays1, 4))

            for i, week in enumerate(weeks):
                self.Tasks[i] = Task(task_id=i,
                                     experiment=experiment,
                                     class_list=cls_list,
                                     split=self._split,
                                     patients=self.patients,
                                     cumm_cls=cumm_cls.copy(),
                                     modelname=str(experiment) + "-" + str(i),
                                     prev_modelname=str(
                                         experiment) + "-" + str(i - 1) if i > 0 else "",
                                     arrays=week)

            print("sdf")

        else:
            raise RuntimeError("Experiment not implemented")

    def load(self):
        print("Loading patient lists and labels", end="... ", flush=True)
        with open("./../ContMIL-main/dataset/split.dat", "rb") as f:
            split = pickle.load(f)
        with open("./../ContMIL-main/dataset/patients.dat", "rb") as f:
            patients = pickle.load(f)
        print("[done]")
        return split, patients

    #load small data 
    # def load(self):
    #     print("Loading patient lists and labels", end="... ", flush=True)
    #     with open("./../ContMIL-main/dataset/split-small.dat", "rb") as f:
    #         split = pickle.load(f)
    #     with open("./../ContMIL-main/dataset/patients-small.dat", "rb") as f:
    #         patients = pickle.load(f)
    #     print("[done]")
    #     return split, patients

    def __len__(self):
        return len(self.Tasks)

    def __getitem__(self, index):
        if index > len(self.Tasks) - 1:
            raise IndexError
        return self.Tasks[index]

    @staticmethod
    def generate_label_file():
        print("labels.dat not found, generating it....")
        data = {}
        for dataset in ["aml", "pbc", "mll"]:
            print("loading ", dataset, "...", end="", flush=True)
            with gzip.open("dataset/" + dataset + "-small.pkl.gz", "rb") as f:
                ndata = pickle.load(f)
            for d in ndata:
                ndata[d]["dataset"] = dataset
            data = {**data, **ndata}
            print("[done]", flush=True)

        a = [{x: data[x]["label"]} for x in data]
        with open("dataset/reduced_data/label_dict.dat", "wb") as f:
            pickle.dump(a, f)

        print("created the labels.dat file")
        exit()

    @staticmethod
    def process_labels(lbls):
        equivalent_classes = {
            # PBC dataset
            'basophil': 'basophil',
            'eosinophil': 'eosinophil',
            'erythroblast': 'erythroblast',
            'IG': "unknown",  # immature granulocytes,
            'PMY': 'promyelocyte',  # immature granulocytes,
            'MY': 'myelocyte',  # immature granulocytes,
            'MMY': 'metamyelocyte',  # immature granulocytes,
            'lymphocyte': 'lymphocyte_typical',
            'monocyte': 'monocyte',
            'NEUTROPHIL': "unknown",
            'BNE': 'neutrophil_banded',
            'SNE': 'neutrophil_segmented',
            'platelet': "unknown",
            # Cytomorphology dataset
            'BAS': 'basophil',
            'EBO': 'erythroblast',
            'EOS': 'eosinophil',
            'KSC': 'smudge_cell',
            'LYA': 'lymphocyte_atypical',
            'LYT': 'lymphocyte_typical',
            'MMZ': 'metamyelocyte',
            'MOB': 'monocyte',  # monoblast
            'MON': 'monocyte',
            'MYB': 'myelocyte',
            'MYO': 'myeloblast',
            'NGB': 'neutrophil_banded',
            'NGS': 'neutrophil_segmented',
            'PMB': "unknown",
            'PMO': 'promyelocyte',
            # MLL dataset
            '01-NORMO': 'erythroblast',
            '04-LGL': "unknown",  # atypical
            '05-MONO': 'monocyte',
            '08-LYMPH-neo': 'lymphocyte_atypical',
            '09-BASO': 'basophil',
            '10-EOS': 'eosinophil',
            '11-STAB': 'neutrophil_banded',
            '12-LYMPH-reaktiv': 'lymphocyte_atypical',
            '13-MYBL': 'myeloblast',
            '14-LYMPH-typ': 'lymphocyte_typical',
            '15-SEG': 'neutrophil_segmented',
            '16-PLZ': "unknown",
            '17-Kernschatten': 'smudge_cell',
            '18-PMYEL': 'promyelocyte',
            '19-MYEL': 'myelocyte',
            '20-Meta': 'metamyelocyte',
            '21-Haarzelle': "unknown",
            '22-Atyp-PMYEL': "unknown",
        }
        label_map = {
            'basophil': 0,
            'eosinophil': 1,
            'erythroblast': 2,
            'myeloblast': 10,
            'promyelocyte': 3,
            'myelocyte': 4,
            'metamyelocyte': 7,
            'neutrophil_banded': 6,
            'neutrophil_segmented': 5,
            'monocyte': 8,
            'lymphocyte_typical': 9,
            'lymphocyte_atypical': 11,
            'smudge_cell': 12,
        }
        ret = {}
        for lbl in lbls:
            lkey = list(lbl.keys())[0]
            lval = lbl[lkey]
            ret[lkey] = {"original_label": lval, "common_label": equivalent_classes[lval],
                         "label": label_map[equivalent_classes[lval]]}
        return ret
