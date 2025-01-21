import os

experiments = ["1", "2", "3", "4"]
#mths = ["lb", "ewc", "icarl", "ucicarl"]
mths = ["mas"]

if os.path.exists("run.sh"):
    os.remove("run.sh")

f = open("run.template", "r")
sh = open("run.sh", "w")

sh.writelines("#!/bin/bash\n\n")

lines = f.readlines()

for exp in experiments:
    for mth in mths:
        if os.path.exists("sb-" + exp + "-" + mth + ".cmd"):
            os.remove("sb-" + exp + "-" + mth + ".cmd")
for exp in experiments:
    for mth in mths:
        c = lines.copy()
        for i in range(len(c)):
            c[i] = c[i].replace("@mth@", str(mth))
            c[i] = c[i].replace("@exp@", str(exp))


        with open("sb-" + exp + "-" + mth + ".cmd", "w") as g:
            g.writelines(c)
            g.flush()
            g.close()

        sh.writelines("sbatch " + "sb-" + exp + "-" + mth + ".cmd\n")
        # sh.writelines("sleep 0.5\n")

sh.flush()
sh.close()


