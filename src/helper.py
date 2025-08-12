import numpy as np
from src.qbm import QBM
from src.rbm import RBM
import json
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,precision_score, recall_score
import tqdm
from src.drawer import draw_test_dataset
from src.utils import import_dataset, split_dataset_labels, split_data
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def get_model(model_name,training_data, params_file_name=None, weights_file_path=None):
    
    '''
    model_names SA RBM,
    '''
    
    if model_name == "SA":
        
        if params_file_name is None or weights_file_path is None :
            params_file_name = './src/models/SA.json'
            weights_file_path = (
            "./src/models/final_weights_qbm.npz")
    
    elif model_name=="RBM":
        
        params_file_name = './src/models/RBM.json'
        weights_file_path = (
            "./src/models/final_weights_rbm.npz"
        )
    try:
        with open(params_file_name, 'r') as json_file:
            loaded_data = json.load(json_file)
        
    except FileNotFoundError as e:
        print("Caught a FileNotFoundError.")
        raise  # Re-raise the same exception
    except IOError as e:  # General I/O error, could include issues like permission denied
        print("An I/O error occurred:", e)
        raise  # Re-raise the exception

    if model_name=='RBM':
        epochs=loaded_data['epochs']
        n_hidden_nodes=loaded_data['n_hidden_nodes']
        
        momentum_coefficient=loaded_data['momentum_coefficient']

        weight_decay=loaded_data['weight_decay']
        
        seed=loaded_data['seed']
        batch_size=loaded_data['batch_size']
        learning_rate=loaded_data['learning_rate']
        model = RBM(training_data, n_hidden_nodes, CD_K=2, SEED=seed, epochs=epochs,momentum_coefficient=momentum_coefficient,
                  weight_decay=weight_decay,trained=True, quantile=QUANTILE)

        
        
    else:        
        epochs=loaded_data['epochs']
        n_hidden_nodes=loaded_data['n_hidden_nodes']
        solver=loaded_data['solver']
        sample_count=loaded_data['sample_count']
        anneal_steps=loaded_data['anneal_steps']
        beta_eff=loaded_data['beta_eff']
        seed=loaded_data['seed']
        restricted=loaded_data['restricted']
        batch_size=loaded_data['batch_size']
        learning_rate=loaded_data['learning_rate']
        
    
        model=QBM(data=training_data, epochs=epochs, n_hidden_nodes=n_hidden_nodes, seed=seed, weight_csv=None, solver=solver,
                 sample_count=sample_count, anneal_steps=anneal_steps, beta_eff=beta_eff, quantile=0.95, trained=True, restricted=restricted,
                 param_string="", savepoint=weights_file_path)
    
    

    return model


def compare_models(cluster_outlier_energy_data,thresholds_list):
    
    '''
    Prints an energy plot comparing different models.
    Provide data in order SA,RBM and then QBM on Hardware.
    
    Args:

    cluster_outlier_energy_data (list): list of (cluster,outlier) energy tuples for each model.
    thresholds(list): list of energy thresholds for each model
    
    '''
    
    assert len(thresholds_list)==len(cluster_outlier_energy_data), "Cluster-outlier data not equal to number of thresholds provided"
    model_names=['SA','RBM','QBM']
   
    keys=[]
    data={}
    for i, model_data in enumerate(cluster_outlier_energy_data):
        
        data[f"{model_names[i]}"]= [model_data[0],model_data[1]]
       
        

        keys.append(model_names[i])

 



        

    fig = plt.figure()
    fig.suptitle('Point Energies', fontsize=14, fontweight='bold')

    ax = fig.add_subplot()

    xlabels = []
    i = 0
    for _ in keys:
        xlabels.append("cluster")
        xlabels.append("outlier")
        i += 2



    raw_data = []
    index = 0
    box_index = 0
    for key, (cluster, outlier) in data.items():
        
            mean = np.linalg.norm(np.concatenate([cluster,outlier]))
            raw_data.append(cluster*10/mean)
            raw_data.append(outlier*10/mean)
            thresholds_list[index] = thresholds_list[index]*10/mean
            xmin = (box_index)/int(len(keys)*2)
            xmax = (box_index+2)/int(len(keys)*2)
            ax.axhline(y=thresholds_list[index], xmin=xmin, xmax=xmax)
            box_index += 2
            
            index += 1
        
    box = ax.boxplot(raw_data, showfliers=False, showmeans=True, vert=True, patch_artist=True)

    boxes = []
    i = 0
    for _ in keys:
        
        boxes.append(box["boxes"][i])
        i += 2

    colors = ['pink', 'lightblue','red']
    colors = colors[:len(keys)]
    colors = [ele for ele in colors for i in range(2)]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)


    ax.legend(boxes, keys, loc='upper right')

    ax.set_ylabel('Energy')
    ymin=-0.8
    ymax=0.0
    plt.ylim(ymin, ymax)
    ax.set_xticks(range(1,2*len(keys)+1))
    ax.set_xticklabels(xlabels, fontsize=8)
    plt.show()




def evaluate_rbm(rbm,testing_dataset,cluster,quantile=0.95):
    
    '''
    Evaluates the RBM on the testing dataset.
    Parameters:
    rbm : RBM instance
    testing_dataset 
    cluster : The number of clusters in the dataset
    quantile
    
    Returns:
    
    A list of Cluster and Outlier energies. 
    '''
    
    
    outliers = RBM.get_binary_outliers(
    dataset=testing_dataset, outlier_index=cluster)
    points = RBM.get_binary_cluster_points(
    dataset=testing_dataset, cluster_index=cluster-1)

    
    testing_data, testing_labels = split_dataset_labels(testing_dataset)
    tensor_testing_data = torch.from_numpy(RBM.binary_encode_data(testing_data)[0])
    tensor_testing_labels = torch.from_numpy(testing_labels)

    outlier_energy = []

    for outlier in outliers:
        outlier = torch.from_numpy(np.reshape(outlier, (1, rbm.num_visible)))
        outlier_energy.append(rbm.free_energy(outlier).cpu().numpy().tolist())
    
    outlier_energy = np.array(outlier_energy)
    
    cluster_point_energy = []
    
    for point in points:
        point = torch.from_numpy(np.reshape(point, (1, rbm.num_visible)))
        cluster_point_energy.append(rbm.free_energy(point).cpu().numpy().tolist())
    
    cluster_point_energy = np.array(cluster_point_energy)
    
    o = outlier_energy.reshape((outlier_energy.shape[0]))
    c = cluster_point_energy.reshape((cluster_point_energy.shape[0]))
    
    #RBM.plot_energy_diff([o, c], rbm.outlier_threshold, "rbm_energies.pdf")
    
    #RBM.plot_hist(c, o, rbm.outlier_threshold, "rbm_hist.pdf")
    
    
    ########## OUTLIER CLASSIFICATION ##########
    print('Outlier classification...')
    
    predict_points = np.zeros(len(tensor_testing_data), dtype=int)
    
    for index, point in enumerate(tensor_testing_data.split(1),0):
        point = point.view(1, rbm.num_visible)
        predict_points[index], _ = rbm.predict_point_as_outlier(point)
    
    true_points = np.where(testing_labels < cluster, 0, 1)
    accuracy, precision, recall = accuracy_score(true_points, predict_points), precision_score(true_points,predict_points), recall_score(true_points, predict_points)
    f1 = f1_score(true_points, predict_points)
    tn, fp, fn, tp = confusion_matrix(true_points, predict_points, labels=[0, 1]).ravel()
   
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}, Num True Negative: {tn}, Num False Negative: {fn}, Num True           Positive: {tp}, Num False Positive: {fp}')


    final_metrics=[accuracy,precision,recall,f1,tn,fp,fn,tp]
    
    return [c,o]

    




def evaluate_qbm(qbm,testing_dataset,cluster,plot=False,quantile=0.95):

    '''
    Evaluates the QBM on the testing dataset.
    Parameters:
    qbm : QBM instance
    testing_dataset 
    cluster : The number of clusters in the dataset
    quantile
    
    Returns:
    
    A list of Cluster and Outlier energies. 
    
    '''
    #training_data=numpy.expand_dims(training_data[:,0],axis=1)
    outliers = qbm.get_binary_outliers(
    dataset=testing_dataset, outlier_index=cluster)

    #outliers=numpy.expand_dims(outliers[:,0],axis=1)
    

    points = qbm.get_binary_cluster_points(dataset=testing_dataset, cluster_index=cluster-1)

    #points=numpy.expand_dims(points[:,0],axis=1)
    #print(points)
    predict_points_cluster = np.zeros(len(points), dtype=int)
    predict_points_outliers = np.zeros(len(outliers), dtype=int)
    qbm.calculate_outlier_threshold(quantile)
    print("Outlier threshold: ", qbm.outlier_threshold)
    print("Calculate outlier Energy")
    
    testing_data, testing_labels = split_dataset_labels(testing_dataset)
#testing_data=numpy.expand_dims(testing_data[:,0],axis=1)

    outlier_energy = []
    for index, outlier in enumerate(tqdm(outliers), 0):
        outlier = np.reshape(outlier, (qbm.dim_input))
        predict_points_outliers[index], this_outlier_energy = qbm.predict_point_as_outlier(
            outlier)
        outlier_energy.append(this_outlier_energy)
    outlier_energy = np.array(outlier_energy)

    o = outlier_energy.reshape((outlier_energy.shape[0]))

    print("Calculate cluster energy")
    cluster_point_energy = []
    
    for index, point in enumerate(tqdm(points), 0):
        point = np.reshape(point, (qbm.dim_input))
        predict_points_cluster[index], this_cluster_point_energy = qbm.predict_point_as_outlier(
        point)
        cluster_point_energy.append(this_cluster_point_energy)
    cluster_point_energy = np.array(cluster_point_energy)

    c = cluster_point_energy.reshape((cluster_point_energy.shape[0]))

    title='test'
#qbmqsp.src.utils.save_output(title="cluster_" + title, object=c)
#QBM.plot_energy_diff([o, c], qbm.outlier_threshold, title + ".pdf")

#QBM.plot_hist(c, o, qbm.outlier_threshold, "qbm_hist" + ".pdf")

########## OUTLIER CLASSIFICATION ##########
    print('Outlier classification: Results...')
    predict_points = np.concatenate(
        (predict_points_cluster, predict_points_outliers))

    #print("Predicted points test: ", predict_points)

    true_points = np.concatenate(
        (np.zeros_like(cluster_point_energy), np.ones_like(outlier_energy)))

    accuracy, precision, recall = accuracy_score(true_points, predict_points), precision_score(
        true_points, predict_points), recall_score(true_points, predict_points)
    f1 = f1_score(true_points, predict_points)
    tn, fp, fn, tp = confusion_matrix(
        true_points, predict_points, labels=[0, 1]).ravel()
    if plot==True:
        print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}, \nNum True Negative: {tn}, Num False Negative: {fn}, Num True         Positive: {tp}, Num False Positive: {fp}')

#print(f'Wallclock time: {(end-start):.2f} seconds')
        lab=cluster-1
        print("Outlier threshold: ", qbm.outlier_threshold)
        print("Average clusterpoint energy: ", np.average(cluster_point_energy))
        print("Outlier energy: ", outlier_energy)

    if plot==True:
        draw_test_dataset(testing_dataset,cluster)

        # Actual plotting
        # Display the plot
        fig = plt.figure(0)
        fig.suptitle('Point Energy', fontsize=14, fontweight='bold')

        ax = fig.add_subplot()
        ax.boxplot([o,c], showfliers=False, showmeans=True)
        ax.set_xticklabels(['outlier', 'cluster points'], fontsize=8)

        ax.set_ylabel('Energy')

        plt.axhline(qbm.outlier_threshold)

        plt.plot([], [], '-', linewidth=1, color='orange', label='median')
        plt.plot([], [], '^', linewidth=1, color='green', label='mean')
        plt.legend()

        

    
    
    #plt.title('Predicted Points')
    #sns.scatterplot(x=testing_data[:,0],y=testing_data[:,1], hue=predict_points,palette='coolwarm')
    
    return [c,o]

