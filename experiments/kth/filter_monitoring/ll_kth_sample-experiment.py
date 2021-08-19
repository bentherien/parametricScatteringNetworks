import os
import sys
sys.path.append(str(os.getcwd()))

from parametricSN.utils.helpers import experiments_cli, experiments_mpCommands, logComparison

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
EPOCHS = 100
RUNS_PER_SEED = 1
TOTALRUNS = 4
SCHEDULER = "OneCycleLR"
AUGMENT = "original-cifar"
MODEL = 'linear_layer'
MODEL_LOSS = 'cross-entropy'
SCATT_PARAM_DISTANCE = 1

if __name__ == '__main__':
    PYTHON, DATA_ARG = experiments_cli()


    for EPOCHS in [1000,2000,4000]:
        commands = []
        for SEED in [1390666426]:#,432857963,1378328753,1118756524]:
            for sample in ['d']:#, 'c', 'b', 'a']:
                for INIT, LEARNABLE in [("Random",1),("Tight-Frame",1)]:

                    command = "{} {} run-train -oname {} -olr {} -gseed {} -sl {} -me {} -odivf {} -sip {}  -os {} -daug {} -en {} -mname {} -pf {} -dsam {} -mloss {} -spd {} {}".format(
                    PYTHON,RUN_FILE,OPTIM,LR,SEED,LEARNABLE,EPOCHS,DF,INIT,SCHEDULER,AUGMENT,mlflow_exp_name, MODEL, PARAMS_FILE, sample, MODEL_LOSS,DATA_ARG,SCATT_PARAM_DISTANCE)

                    commands.append(command)
            

        experiments_mpCommands(
            processBatchSize=PROCESS_BATCH_SIZE,
            commands=commands
        )


        logComparison(mlflow_exp_name)
    
    exit(0)

