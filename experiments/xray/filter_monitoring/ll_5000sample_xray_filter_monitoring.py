import os
import sys
sys.path.append(str(os.getcwd()))

from parametricSN.utils.helpers import experiments_cli, experiments_mpCommands, logComparison

mlflow_exp_name = os.path.basename(__file__)
PROCESS_BATCH_SIZE = 1
PARAMS_FILE = "parameters_xray.yml"
RUN_FILE = "parametricSN/main.py"
OPTIM = "sgd"
LR = 0.01
LRS = 0.01
LRO = 0.01
LRMAX = 0.01
DF = 25
LEARNABLE = 1
EPOCHS = 400
INIT = "Tight-Frame"
RUNS_PER_SEED = 10
TOTALRUNS = 2 * RUNS_PER_SEED
SCHEDULER = "OneCycleLR"
TRAIN_SAMPLE_NUM = 5000
TRAIN_BATCH_SIZE = 128
AUGMENT = "original-cifar"
SECOND_ORDER = 0
MODEL="linear_layer"
SCATT_PARAM_DISTANCE = 1


if __name__ == '__main__':
    PYTHON, DATA_ARG = experiments_cli()

    for EPOCHS in [100,200,400]:
        commands = []

        for SEED in [22942091]:
            for aa in [(1,"Tight-Frame"),(1,"Random")]:
                LEARNABLE, INIT = aa

                args1 = "-daug {} -en {} -pf {} -sso {} -mname {} -spd {} {}".format(
                    AUGMENT,mlflow_exp_name,PARAMS_FILE,SECOND_ORDER,MODEL,SCATT_PARAM_DISTANCE,DATA_ARG)

                args2 = "-oname {} -olr {} -gseed {} -sl {} -me {} -omaxlr {} -odivf {} -sip {} -dtsn {} -dtbs {} -os {}".format(
                    OPTIM,LR,SEED,LEARNABLE,EPOCHS,LRMAX,DF,INIT,TRAIN_SAMPLE_NUM,TRAIN_BATCH_SIZE,SCHEDULER)

                args3 = "-slrs {} -slro {}".format(
                    LRS,LRO)
                
                command = "{} {} run-train {} {} {}".format(
                    PYTHON,RUN_FILE,args1,args2,args3)

                commands.append(command)

        experiments_mpCommands(
            processBatchSize=PROCESS_BATCH_SIZE,
            commands=commands
        )

        logComparison(mlflow_exp_name)

    exit(0)




    