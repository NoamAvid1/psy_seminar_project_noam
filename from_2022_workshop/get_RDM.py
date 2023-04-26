import plotly.express as px
import pandas as pd
import os

dataset_titles = ["Symmetrical Greebles", "Asymmetrical Greebles"]
data_types = ["symmetric", "asymmetric"] # symmetric or asymmetric
net_types = ["objects","faces"]  # objects or faces


# Save RDM mat for each combination of dataset and net type
for i in range(len(data_types)):
  data_type = data_types[i]
  dataset_title_name = dataset_titles[i]
  for net_type in net_types:
    # note that the old folder path is from the "sorted_results" file, for by-families order.
    old_folder_path = fr'/home/ssd_storage/experiments/students/Noam/greebles/{data_type}_greebles_{net_type}/vgg16/results/sorted_results/'   
    new_folder_path = fr'/home/ssd_storage/experiments/students/Noam/greebles/RDM_results/{data_type}_greebles_{net_type}/'
      
    result_files = [f for f in os.listdir(old_folder_path)]
    
    for file_name in result_files:
     old_file_path = os.path.join(old_folder_path, file_name)
     layer_name = file_name[:-4]
     new_file_name = layer_name + "_"+ data_type + "_"+ net_type + "_RDM" + ".html"
     new_file_path = os.path.join(new_folder_path, new_file_name)
     file_data = pd.read_csv(old_file_path)
     
     fig = px.imshow(file_data,  labels=dict(color="Disimilarity"))
     
     fig.update_xaxes(side="top")
     fig.update_layout(
         width = 650,
         height = 650,
         # updating the X axis titles and tick labels:
         xaxis=dict(
             tickvals = [i*32+10 for i in range(5)],   # tick labels distances
             ticktext = ['Family 1', 'Family 2', 'Family 3', 'Family 4', 'Family 5'],
             title_text= f"RDM for {dataset_title_name}<br>net type = {net_type}, layer = {layer_name}" ),
         # updating the y axis titles and tick labels:  
         yaxis=dict(
             tickvals = [i*32 for i in range(5)],
             ticktext = ['Family 1', 'Family 2', 'Family 3', 'Family 4', 'Family 5']),             
         #xaxis_nticks=5
       )
     fig.update_xaxes(automargin=True)
     # comment this out if you want program to open csv in firefox:
     #fig.show()    
     fig.write_html(new_file_path)
    
