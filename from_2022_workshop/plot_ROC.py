import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import os 


dataset_titles = ["Symmetrical Greebles", "Asymmetrical Greebles"]
data_types = ["symmetric", "asymmetric"] # symmetric or asymmetric
net_types = ["objects","faces"]  # objects or faces

for i in range(len(data_types)):
  data_type = data_types[i]
  dataset_title_name = dataset_titles[i]
  for net_type in net_types:
    # note that the old folder path is from the "sorted_results" file, for by-families order.
    old_folder_path = fr'/home/ssd_storage/experiments/students/Noam/greebles/{data_type}_greebles_{net_type}/vgg16/results/sorted_results/'   
    new_folder_path = fr'/home/ssd_storage/experiments/students/Noam/greebles/ROC_results/{data_type}_greebles_{net_type}/'
    result_files = [f for f in os.listdir(old_folder_path)]
    
    for file_name in result_files:
     old_file_path = os.path.join(old_folder_path, file_name)
     layer_name = file_name[:-4]
     new_file_name = layer_name + "_"+ data_type + "_"+ net_type + "_ROC" + ".html"
     new_file_path = os.path.join(new_folder_path, new_file_name)
     file_data = pd.read_csv(old_file_path)
     
     # Setting up data arrays for plotting:
     data_arr = np.array(file_data)[ : ,1:]   # array of the csv data (not including title)
     data_vector_shape = (data_arr.shape[0] * data_arr.shape[1] - data_arr.shape[0],)  # desired vector size is (160X160) - 160, Since we will delete the main diagonal  
     # create vectors for roc plotting:
     same_pair_mat = np.ones((2,2))
     y_test = np.kron(np.eye((data_arr.shape[0]//2), dtype=int), same_pair_mat)   # y_test = block diagonal matrix, each block is 2X2 matrix of "1"s.
     # delete main diagonal (not comparing a picture to itself!):
     y_test = y_test[~np.eye(y_test.shape[0],dtype=bool)].reshape(y_test.shape[0],-1)
     # reshape y_test into a vector:
     y_test = y_test.reshape(*data_vector_shape)
     arr_norm = np.linalg.norm(data_arr) # compute std
     # noramlized the distances and get the "similarity" score:
     probas = (1-data_arr/arr_norm)
     probas = probas[~np.eye(probas.shape[0],dtype=bool)].reshape(probas.shape[0],-1)
     # reshape probas into a vector:
     probas = probas.reshape(*data_vector_shape)
     
     # Compute ROC curve and area the curve:
     fpr, tpr, thresholds = roc_curve(y_test, probas)
     roc_auc = auc(fpr, tpr)
     ### print("Area under the ROC curve : %f" % roc_auc)  
     
     # Plot ROC curve:
     fig = px.area(
         x=fpr, y=tpr,
         title=f'ROC Curve for {dataset_title_name} <br>net type = {net_type}, layer= {layer_name}, AUC={auc(fpr, tpr):.4f}',
         labels=dict(x='False Positive Rate', y='True Positive Rate'),
         width=700, height=500
     )
     fig.add_shape(
         type='line', line=dict(dash='dash'),
         x0=0, x1=1, y0=0, y1=1
     )
     
     fig.update_yaxes(scaleanchor="x", scaleratio=1)
     fig.update_xaxes(constrain='domain')
     #fig.show()
     fig.write_html(new_file_path)
     
     
     




