class Task:
    def __init__(self, task_id, experiment, class_list, patients, split, cumm_cls, modelname, prev_modelname,
                 arrays=""):
        self.task_id = task_id
        self.experiment = experiment
        self.class_list = class_list
        self.cumm_cls = cumm_cls
        self.modelname = modelname
        self.prev_modelname = prev_modelname
        self.patients = {}
        self.arrays = arrays

        if experiment == 1:
            for ti, t in enumerate(["trainset", "testset"]):
                self.patients[t] = {}
                patientlist = split[ti]
                for p in patientlist:
                    r, lbl = self.convert_label(patients[p]["label_conv"])
                    if r and lbl in self.class_list:
                        self.patients[t][p] = patients[p]
                        self.patients[t][p]["label"] = lbl
        elif experiment == 2:
            for ti, t in enumerate(["trainset", "testset"]):
                self.patients[t] = {}
                patientlist = split[ti]
                for p in patientlist:
                    if int(patients[p]["array_id"].split("_")[1]) in arrays[ti]:
                        r, lbl = self.convert_label(patients[p]["label_conv"])
                        if r and lbl in self.class_list:
                            self.patients[t][p] = patients[p]
                            self.patients[t][p]["label"] = lbl

    def __str__(self):
        ret = "exp: " + str(self.experiment) + " - task: " + str(self.task_id) + str(self.class_list) + str(
            len(self.patients["trainset"])) + "," + str(len(self.patients["testset"]))
        return ret

    def print(self):
        tot = 0
        for ti, t in enumerate(["trainset", "testset"]):
            print(t)
            for cls in self.class_list:
                c = len([p for p in self.patients[t] if self.patients[t][p]["label"] == cls])
                tot += c
                print(cls, ": ", c)

    @staticmethod
    def convert_label(lbl):
        label_converter = {
            # 'Reactive changes': 0,
            'AML': 1,
            'CML': 2,
            'No evidence': 0,
            'CMML': 6,
            'CLL': 4,
            'MPN': 5,
            'MDS': 3,
            "Lymphoma": 7
        }
        if lbl in label_converter.keys():
            return True, label_converter[lbl]
        else:
            return False, ""
