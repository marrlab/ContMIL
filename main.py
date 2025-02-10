import sys
from ContinualLearner import ContinualLearner
from TaskManager import TaskManager
from utils.utils import loaddata

# bebin data asan bage khali dare ya na, chon random kar nemikone

if __name__ == '__main__':
    experiment = int(sys.argv[2])
    method = sys.argv[1]
    # experiment = 1
    # method = 'lb'
    tm = TaskManager(experiment=experiment)
    data = loaddata()


    # for i, task in enumerate(tm):
    #     if i == 0:
    #         continue
    #     print(task)
    #     cl = ContinualLearner(task, method)
    #     # cl = ContinualLearner(tm[1], method)
    #     cl.train_task(data)
    #     cl.after_task(tm, data)

    for task in tm:
        print(task)
        cl = ContinualLearner(task, method)
        # cl = ContinualLearner(tm[1], method)
        cl.train_task(data)
        cl.after_task(tm, data)
