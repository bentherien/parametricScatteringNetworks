import os
import sys
sys.path.append(str(os.getcwd()))

from parametricSN.utils.helpers import experiments_cli, experiments_mpCommands

def getExpName(file):
    return "-".join(file.split("/"))
    
mlflow_exp_name = getExpName(__file__)

PROCESS_BATCH_SIZE = 2

RUN_FILE = "parametricSN/main.py"
OPTIM = "sgd"
LR = 0.1
LRS = 0.1
LRO = 0.1
LRMAX = 0.06
DF = 25
LEARNABLE = 1
EPOCHS = 5000
INIT = "Tight-Frame"
RUNS_PER_SEED = 10
TOTALRUNS = 2 * RUNS_PER_SEED
SCHEDULER = "OneCycleLR"
TRAIN_SAMPLE_NUM = 100
TRAIN_BATCH_SIZE = 128
AUGMENT = "autoaugment"

# 0.156016M

if __name__ == '__main__':
    PYTHON, DATA_ARG = experiments_cli()

    commands = []

    for SEED in [491659600]:
        for aa in [(1,"Tight-Frame"),(0,"Tight-Frame")]:
            LEARNABLE, INIT = aa

            command = "{} {} run-train -oname {} -olr {} -gseed {} -sl {} -me {} -omaxlr {} -odivf {} -sip {} -dtsn {} -dtbs {} -os {} -daug {} -en {} {}".format(
                PYTHON,RUN_FILE,OPTIM,LR,SEED,LEARNABLE,EPOCHS,LRMAX,DF,INIT,TRAIN_SAMPLE_NUM,TRAIN_BATCH_SIZE,SCHEDULER,AUGMENT,mlflow_exp_name,DATA_ARG)

            commands.append(command)
    
    experiments_mpCommands(
        processBatchSize=PROCESS_BATCH_SIZE,
        commands=commands
    )

   