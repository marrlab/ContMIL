import sys
from ContinualLearner import ContinualLearner
from TaskManager import TaskManager
from utils.utils import loaddata


if __name__ == '__main__':
    # experiment = int(sys.argv[2])
    experiment = 1
    # method = sys.argv[1]
    method = 'der'
    tm = TaskManager(experiment=experiment)
    data = loaddata(True)

    for i, task in enumerate(tm):
        if i == 0:
            continue
        print(task)
        cl = ContinualLearner(task, method)
        # cl = ContinualLearner(tm[1], method)
        cl.train_task(data)
        cl.after_task(tm, data)
