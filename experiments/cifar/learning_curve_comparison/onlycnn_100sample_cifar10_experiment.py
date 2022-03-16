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
LRMAX = 0.1
DF = 25
THREE_PHASE = 1
EPOCHS = 3000
RUNS_PER_SEED = 10
SCHEDULER = "OneCycleLR"
TEST_BATCH_SIZE = 256
TRAIN_SAMPLE_NUM = 100
TRAIN_BATCH_SIZE = 128
AUGMENT = "autoaugment"
MODEL = "cnn"
PHASE_ENDS = " ".join(["100","200"])
MODEL_WIDTH = 8
SCATT_ARCH = 'identity'
MODEL_LOSS = 'cross-entropy'
SCATT_LRMAX = 0.2
SCATT_DF = 25
SCATT_THREE_PHASE = 1


if __name__ == '__main__':
    PYTHON, DATA_ARG = experiments_cli()

    commands = []


    for SEED in [207715039]: #491659600,207715039,737523103,493572006,827192296,877498678,1103100946,1210393663,1277404878,1377264326]:
        args1 = "-oname {} -olr {} -gseed {} -me {} -omaxlr {} -odivf {} -dtsn {}".format(
            OPTIM,LR,SEED,EPOCHS,LRMAX,DF,TRAIN_SAMPLE_NUM
        )

        args2 = "-os {} -daug {} -en {} -dtbs {} -mname {} -ope {}".format(
            SCHEDULER,AUGMENT,mlflow_exp_name,TRAIN_BATCH_SIZE,MODEL,PHASE_ENDS
        )

        args3 = "-smaxlr {} -sdivf {} -stp {} -mloss {} -sa {} -mw {} -dtstbs {}".format(
            SCATT_LRMAX,SCATT_DF,SCATT_THREE_PHASE,MODEL_LOSS,SCATT_ARCH,MODEL_WIDTH,TEST_BATCH_SIZE
        )

        command = "{} {} run-train {} {} {} {}".format(
            PYTHON,RUN_FILE,args1,args2,args3,DATA_ARG)

        commands.append(command)
    

    experiments_mpCommands(
        processBatchSize=PROCESS_BATCH_SIZE,
        commands=commands
    )







