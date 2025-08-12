import argparse
import os
import random
import time

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from tqdm import tqdm

import src.utils
from src.qbm import QBM
from src.utils import import_dataset, split_dataset_labels, split_data

from functools import partial
import wandb

DATASET_PATH = 'src/datasets/l_o6_c10_d3_p1000_2.npy'
CLUSTER = 10


def perform_run(seed=77, n_hidden_nodes=128, solver="SA", sample_count=20,
         anneal=1000, beta_eff=1, epochs=3, batch_size=10, learning_rate=0.005,
         restricted=True, save="", name=""):

    '''
    Performs training and testing of the QBM for one set of parameters.
    '''    
    
    print("Start")

    random.seed(seed)
    np.random.seed(seed)
    print("Seed is " + str(seed))

    start = time.time()

    data = import_dataset(DATASET_PATH)
    training_dataset, testing_dataset = split_data(data, CLUSTER)
    training_data, training_labels = split_dataset_labels(training_dataset)

    dataset_name = os.path.splitext(DATASET_PATH.split("/")[-1])[0]
    param_string = "_se" + str(seed) + "_h" + str(n_hidden_nodes) + "_sol" + solver + "_sc" + str(sample_count) + "_b" + str(
        beta_eff) + "_e" + str(epochs) + "_l" + str(learning_rate) + "_r" + str(restricted) + "_data" + dataset_name + "_n_" + name

    # create DQBM
    dqbm = QBM(seed=seed, data=training_data, epochs=epochs,
               n_hidden_nodes=n_hidden_nodes, solver=solver,
               restricted=restricted, sample_count=sample_count,
               anneal_steps=anneal, beta_eff=beta_eff,
               param_string=param_string, savepoint=save)
    # train
    print('Training QBM...')
    dqbm.train_model(batch_size=batch_size, learning_rate=learning_rate)

    # test
    testing_data, testing_labels = split_dataset_labels(testing_dataset)
    encoded_data_test, bits_input_vector, num_features = QBM.binary_encode_data(
        testing_data, use_folding=True)

    ########## EXTRACT FEATURES ##########
    #print('Extracting features... Skipped')
    #train_features = dqbm.get_hidden_features(dqbm.encoded_data)
    #test_features = dqbm.get_hidden_features(encoded_data_test)

    ########## CLASSIFICATION ##########
    #print('Classifying hidden activations as test... Skipped')

    #sag = LogisticRegression(solver="sag", max_iter=500)
    #sag.fit(train_features, training_labels)
    #predictions = sag.predict(test_features)

    #print('Result: {0:.2%}'.format(sum(predictions == testing_labels) / testing_labels.shape[0]))
    
    #print("Result: Skipped")

    ########## ENERGY COMPARISON ##########
    print('Energy comparison & Outlier classification...')

    outliers = QBM.get_binary_outliers(
        dataset=testing_dataset, outlier_index=CLUSTER)
    predict_points_outliers = np.zeros(len(outliers), dtype=int)
    points = QBM.get_binary_cluster_points(
        dataset=testing_dataset, cluster_index=CLUSTER - 1)
    predict_points_cluster = np.zeros(len(points), dtype=int)

    print("Outlier threshold: ", dqbm.outlier_threshold)
    print("Calculate outlier Energy")
    outlier_energy = []
    for index, outlier in enumerate(tqdm(outliers), 0):
        outlier = np.reshape(outlier, (dqbm.dim_input))
        predict_points_outliers[index], this_outlier_energy = dqbm.predict_point_as_outlier(
            outlier)
        outlier_energy.append(this_outlier_energy)
    outlier_energy = np.array(outlier_energy)

    o = outlier_energy.reshape((outlier_energy.shape[0]))
    title = "qbm_energies" + dqbm.paramstring
    src.utils.save_output(title="outlier_" + title, object=o)
    print("Calculate cluster energy")
    cluster_point_energy = []
    for index, point in enumerate(tqdm(points), 0):
        point = np.reshape(point, (dqbm.dim_input))
        predict_points_cluster[index], this_cluster_point_energy = dqbm.predict_point_as_outlier(
            point)
        cluster_point_energy.append(this_cluster_point_energy)
    cluster_point_energy = np.array(cluster_point_energy)

    c = cluster_point_energy.reshape((cluster_point_energy.shape[0]))

    src.utils.save_output(title="cluster_" + title, object=c)
    #QBM.plot_energy_diff([o, c], dqbm.outlier_threshold, title + ".pdf")

    #QBM.plot_hist(c, o, dqbm.outlier_threshold, "qbm_hist" + param_string + ".pdf")

    ########## OUTLIER CLASSIFICATION ##########
    print('Outlier classification: Results...')
    predict_points = np.concatenate(
        (predict_points_cluster, predict_points_outliers))

    print("Predicted points test: ", predict_points)

    true_points = np.concatenate(
        (np.zeros_like(cluster_point_energy), np.ones_like(outlier_energy)))

    accuracy, precision, recall = accuracy_score(true_points, predict_points), precision_score(
        true_points, predict_points), recall_score(true_points, predict_points)
    f1 = f1_score(true_points, predict_points)
    tn, fp, fn, tp = confusion_matrix(
        true_points, predict_points, labels=[0, 1]).ravel()

    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}, \nNum True Negative: {tn}, Num False Negative: {fn}, Num True Positive: {tp}, Num False Positive: {fp}')

    end = time.time()
    print(f'Wallclock time: {(end-start):.2f} seconds')

    print("Outlier threshold: ", dqbm.outlier_threshold)
    print("Average clusterpoint energy: ", np.average(cluster_point_energy))
    print("Outlier energy: ", outlier_energy)

    final_metrics=[[accuracy],[precision],[recall],[f1],[tn],[fp],[fn],[tp]]  # Total 8 metrics
    
    return final_metrics

def get_averages(list_of_lists):
    array_of_arrays = np.array(list_of_lists)
    averages = np.mean(array_of_arrays, axis=0)
    return averages


def configure_hyperparams(run):
    '''
    Sets the hyperparameters for a given run. Both with
    and without hyperparameter optimisation.
    '''
    
    global BATCH_SIZE
    global EPOCHS
    
    global ANNEAL_STEPS
    global N_HIDDEN_UNITS
    global SAMPLE_COUNT
    global BETA_EFF
    global LEARNING_RATE
    #global BASEPATH_RECORD
    global RESTRICTED
    global PARAM_STRING
    #global ARGUMENTS

    if run:
        
        config_defaults = {'batchsize': args.batchsize, 'epochs': args.epochs,'hnodes': 4, 'solver':args.solver,
                           'sample_count': args.sample_count, 'beta_eff': args.beta_eff,
                           'restricted':args.restricted, 'learning_rate':args.learning_rate,
                           'anneal_steps':args.anneal_steps}
    
        run.config.setdefaults(config_defaults)

        # set hyperparameters from sweep
        BATCH_SIZE = wandb.config.batchsize
        EPOCHS = wandb.config.epochs
        LEARNING_RATE=wandb.config.learning_rate
        # define network parameters for dqbm
        N_HIDDEN_UNITS = wandb.config.hnodes
        SAMPLE_COUNT = wandb.config.sample_count
        BETA_EFF = wandb.config.beta_eff
        SOLVER=wandb.config.solver
        RESTRICTED=wandb.config.restricted
        ANNEAL_STEPS=wandb.config.anneal_steps
    else:
        # set hyperparameters
        BATCH_SIZE = args.batchsize
        EPOCHS = args.epochs
        LEARNING_RATE=args.learning_rate

        # define network parameters for dqbm
        N_HIDDEN_UNITS = args.hnodes
        SOLVER=args.solver
        SAMPLE_COUNT = args.sample_count
        BETA_EFF = args.beta_eff
        RESTRICTED=args.restricted
        ANNEAL_STEPS=args.anneal_steps
     # params and identifier-strings independent of using hyperparameter
    # optimization or not
    #BASEPATH_RECORD = get_basepath("training", DATASET)
    
    #ARGUMENTS = [BATCH_SIZE, EPOCHS, ACCURACY_INTERVAL,
             #   N_HIDDEN_UNITS, SAMPLE_COUNT, ANNEAL_STEPS, BETA_EFF, TESTSIZE]
    
    PARAM_STRING = "_h" + str(N_HIDDEN_UNITS) + "_sol" + SOLVER + "_sc" + str(SAMPLE_COUNT) + "_b" + str(
        BETA_EFF)+"_lr"+str(LEARNING_RATE) + "_e" + str(EPOCHS) + "_r" + str(RESTRICTED)
    
    
    return PARAM_STRING


def main(args):

    
    ANNEAL_STEPS = args.anneal_steps  # TODO: What to do in D-Wave case?
    SOLVER = args.solver
    

    # set testsize for hyperparameter-tuning
    TESTSIZE = args.test_size
    NUM_RUNS = args.n_runs

    # start run
    if HYPERPARAM_OPT:
        run = wandb.init(reinit=True
                         , group=SWEEP_ID
                         )
    else:
        run = None

    param_string_for_run=configure_hyperparams(run)
    
    #is_first_run = args.first_run

    
    


    
    if HYPERPARAM_OPT:
        run.name = param_string_for_run
        
        print(f"Run Name: {run.name}\n")
        
        
        num_metrics=8   #number of metric returned by the perform_run function
        metrics_for_all_seeds=[[] for i in range(num_metrics)]
    
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 21, 22,
             23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 54, 55, 56, 58,
             60, 61, 62, 63, 64, 65, 66, 70, 71, 72, 73, 74, 75, 76, 77, 78,
             79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 92, 93, 96, 97, 98, 99]

    NUM_RUNS=2
    for run_num in range(NUM_RUNS):
        # determine seed
        seed = random.choice(seeds)
        print("seed: " + str(seed))
        
        PARAM_STRING_SEED = param_string_for_run + "_seed_" + str(seed)
    
        #files_for_this_run = create_or_find_files(is_first_run, seed)
        #is_first_run = 'False'
    
        
        metrics=perform_run(epochs=EPOCHS, n_hidden_nodes=N_HIDDEN_UNITS    ,
            batch_size=BATCH_SIZE,anneal=ANNEAL_STEPS,beta_eff=BETA_EFF,
            learning_rate=LEARNING_RATE,  solver=SOLVER, restricted=RESTRICTED,
            seed=seed, save=args.savepoint, name=args.name)
        
        

        if HYPERPARAM_OPT:
            # get combined metric of accuracy and auc-roc-score
            #combined_metrics = get_combined_metrics(metrics[0], metrics[5])
            #metrics.append(combined_metrics)
            
             for metric_index in range(len(metrics)):
                metrics_for_all_seeds[metric_index].append(metrics[metric_index])

        print(f"Training for seed {seed} done.\n")
        seeds.remove(seed)
    
    
    if HYPERPARAM_OPT:
        #print(metrics_for_all_seeds)
        for metric_index in range(len(metrics_for_all_seeds)):
            metrics_for_all_seeds[metric_index] = get_averages(metrics_for_all_seeds[metric_index])
        #print(metrics_for_all_seeds)
        
        
        #for i in range(len(metrics_for_all_seeds[0])):
        wandb.log({
                       'accuracy': metrics_for_all_seeds[0],
                   'precision': metrics_for_all_seeds[1], 'recall': metrics_for_all_seeds[2],
                   'f1_score': metrics_for_all_seeds[3], 'true_negatives': metrics_for_all_seeds[4],
                   'false_positives': metrics_for_all_seeds[5],
                       'false_negatives': metrics_for_all_seeds[6],
                   'true_positives': metrics_for_all_seeds[7]}
                  )
        run.finish()
    print("Run done.")
#print("Seeds from list left: ", seeds)
         


from start_sweep import sweep_configuration


if __name__ == '__main__':
    

    
    
    parser = argparse.ArgumentParser(
        description='Generate clustered datasets with outliers.')
   
    

    
        
    parser.add_argument('-hn', '--hnodes',
                        metavar='INT',
                        help='Amount of hidden units for RBM model',
                        default=82,  # best fit for dataset l_o7_c5_d3_p200_v1
                        type=int)

    parser.add_argument('-e', '--epochs',
                        metavar='INT',
                        help='Epochs for training',
                        default=13,  # best fit for dataset l_o7_c5_d3_p200_v1
                        type=int)

    parser.add_argument('-b', '--batchsize',
                        metavar='INT',
                        help='Batchsize for training',
                        default=10,  # best fit for dataset l_o7_c5_d3_p200_v1
                        type=int)

    
   
    # random seed for reproducibility of classical (SA) runs
    
    parser.add_argument('-s', '--seed',
                        metavar='INT',
                        help='Seed for RNG',
                        default=77,  # best fit for dataset l_o7_c5_d3_p200_v1
                        type=int)

    
    # solver = 'SA' for local simulator
    # 'QBSolv' for testing UQO without losing coins
    # 'DW_2000Q_6' [max hnodes==94 vnodes==21] or 'Advantage_system4.1' [max hnodes==634 vnodes==21] for D-Wave Quantum Annealer (needs coins)
    # 'FujitsuDAU' for accessing Fujitsu Digital Annealing Unit (needs coins)
    #  when using FujitsuDAU, sample_count must be a value in [16, 128]
    parser.add_argument('--solver',
                        help='Solver, options: \'SA\', \'DW_2000Q_6\', \'Advantage_system4.1\', \'FujitsuDAU\', '
                             '\'MyQLM\', \'BMS\'',
                        default='SA',  # best fit for dataset l_o7_c5_d3_p200_v1
                        type=str)

    parser.add_argument('--savepoint',
                        help='Filepath to numpy file with saved weights to initialize from',
                        default="",
                        type=str)

    parser.add_argument('--name',
                        help='Name for run',
                        default="",
                        type=str)

    parser.add_argument("--sample_count", type=int, default=20)
   
    # number of annealing steps to perform (only relevant for SA)
    
    parser.add_argument("--anneal_steps", type=int, default=1000)
   
    # inverse effective temperature, only relevant for runs on D-Wave
    
    # (What is good default value? 0.5? 1.0? ...?)
    parser.add_argument("--beta_eff", type=float, default=0.5)

   

    parser.add_argument("--learning_rate", type=float, default=0.005)
    
    parser.add_argument("--restricted", type=str, default='true')

    # what fraction of the training dataset to use, can be set to < 1.0 (
    # e.g. 0.2 for tuning hyperparameters)
    parser.add_argument("--test_size", type=float, default=1.0)
    # number of runs with different seeds to execute
    
    parser.add_argument("--n_runs", type=int, default=10)

    # for automized hyperparameter optimization
    
    parser.add_argument("--hyperparam_opt", type=str, default='true')
    parser.add_argument("--n_sweeps", type=int, default=4)
    parser.add_argument("--sweep_id", type=str, default=None)
    
    #Add sweep id , through a separate sweep file instead.
    
    parser.add_argument("--key", type=str, default=None)
    
    args = parser.parse_args()
    print("Running with solver", args.solver)
    
    HYPERPARAM_OPT = args.hyperparam_opt == 'true'
    NUM_SWEEPS = args.n_sweeps

    
    
    if HYPERPARAM_OPT:
        #wandb.login(key=args.key)
        wandb.login()
        
        #SWEEP_ID = args.sweep_id
        
        #sweep_id_path = "mdsg/qbm-anomaly-detection/" + SWEEP_ID
        
        project_name='qbm-anomaly-detection' 
        entity='mdsg'
        # add entity name
        sweep_id_path=wandb.sweep(sweep_configuration,project=project_name,entity=entity)    
        
        sweep_id_path=project_name+'/'+sweep_id_path
        print(sweep_id_path)
        SWEEP_ID=sweep_id_path
        main_with_args = partial(main, args)
        wandb.agent(sweep_id=sweep_id_path, function=main_with_args,
                    count=NUM_SWEEPS)
    else:
        main(args)

