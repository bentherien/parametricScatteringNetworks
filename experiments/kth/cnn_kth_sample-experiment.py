import os
import sys
sys.path.append(str(os.getcwd()))

from parametricSN.utils.helpers import experiments_cli, experiments_mpCommands

mlflow_exp_name = os.path.basename(__file__)

PROCESS_BATCH_SIZE = 2

RUN_FILE = "parametricSN/main.py"

PARAMS_FILE = "parameters_texture.yml"
OPTIM = "sgd"
LR = 0.1
LRS = 0.1
LRO = 0.1
DF = 25
LEARNABLE = 0
INIT = "Tight-Frame"
EPOCHS = 50
RUNS_PER_SEED = 1
TOTALRUNS = 4
SCHEDULER = "OneCycleLR"
AUGMENT = "original-cifar"
MODEL = 'cnn'
MODEL_LOSS = 'cross-entropy'


if __name__ == '__main__':
    PYTHON, DATA_ARG = experiments_cli()

    commands = []
    for SEED in [1390666426,432857963,1378328753,1118756524]:
        for sample in ['d', 'c', 'b', 'a']:
            for x in range(TOTALRUNS):

                LEARNABLE = 0 if LEARNABLE == 1 else 1
                if x % 2 == 0  and x != 0:
                    INIT = "Random" if INIT == "Tight-Frame" else "Tight-Frame"

                command = "{} {} run-train -oname {} -olr {} -gseed {} -sl {} -me {} -odivf {} -sip {}  -os {} -daug {} -en {} -mname {} -pf {} -dsam {} -mloss {} {}".format(
                PYTHON,RUN_FILE,OPTIM,LR,SEED,LEARNABLE,EPOCHS,DF,INIT,SCHEDULER,AUGMENT,mlflow_exp_name, MODEL, PARAMS_FILE, sample, MODEL_LOSS, DATA_ARG)

                commands.append(command)
        
    experiments_mpCommands(
        processBatchSize=PROCESS_BATCH_SIZE,
        commands=commands
    )

   

