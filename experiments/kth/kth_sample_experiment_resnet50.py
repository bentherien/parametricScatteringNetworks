import os
import sys
sys.path.append(str(os.getcwd()))

from parametricSN.utils.helpers import experiments_cli, experiments_mpCommands

mlflow_exp_name = os.path.basename(__file__)
PROCESS_BATCH_SIZE = 1

RUN_FILE = "parametricSN/main.py"
PARAMS_FILE = "parameters_texture.yml"
OPTIM = "sgd"
LR = 0.0001
LRS = 0.1
LRO = 0.1
DF = 25
EPOCHS = 50
RUNS_PER_SEED = 4
TOTALRUNS = 1
SCHEDULER = "OneCycleLR"
AUGMENT = "original-cifar"
ACCUM_STEP_MULTIPLE = 128
TEST_BATCH_SIZE = 128
TRAIN_BATCH_SIZE = 128
SECOND_ORDER = 0
MODEL = 'resnet50'
MODEL_WIDTH = 8
SCATT_ARCH = 'identity'
MODEL_LOSS = 'cross-entropy'

if __name__ == '__main__':
    PYTHON, DATA_ARG = experiments_cli()

    commands = []
    for SEED in [1390666426,432857963,1378328753,1118756524]:
        for sample in ['a', 'b', 'c', 'd']:
            args1 = "-oname {} -olr {} -gseed {} -me {} -odivf {}  -os {} -daug {} -en {} -pf {} -dsam {} {}".format(
                OPTIM,LR,SEED,EPOCHS,DF,SCHEDULER,AUGMENT,mlflow_exp_name,PARAMS_FILE, sample, DATA_ARG
            )
            args2 = "-mw {} -mloss {} -sa {} -dtstbs {} -dtbs {} -mname {} -dasm {}".format(
            MODEL_WIDTH,MODEL_LOSS,SCATT_ARCH,TEST_BATCH_SIZE,TRAIN_BATCH_SIZE,MODEL,ACCUM_STEP_MULTIPLE)
            command = "{} {} run-train {} {}".format(
            PYTHON,RUN_FILE,args1,args2)
            commands.append(command)

    experiments_mpCommands(
        processBatchSize=PROCESS_BATCH_SIZE,
        commands=commands
    )