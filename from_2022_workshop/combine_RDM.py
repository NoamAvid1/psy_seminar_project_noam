import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import os


net_types = ["objects","faces","birds"] 
rotation_state = ["upright", "inverted"]
datasets_info = {"greebles_symmetric":"Symmetrical Greebles", "greebles_asymmetric":"Asymmetrical Greebles", "bird_indiv":"Individual Birds", \
   "bird_species":"Bird Species" , "faces": "Faces","objects":"Objects"}
figs_dict = dict()
results_folder_path = fr'/home/ssd_storage/experiments/students/Noam/results_for_conference_dec21/RDM_results/'


def figures_to_html(figs, filename="dashboard.html"):
    dashboard = open(filename, 'w')
    dashboard.write("<html><head></head><body>" + "\n")
    for fig in figs:
        inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
        dashboard.write(inner_html)
    dashboard.write("</body></html>" + "\n")


# Save RDM mat for each combination of dataset and net type
for dataset in datasets_info:
  dataset_title_name = datasets_info[dataset]
  for net_type in net_types:
    for state in rotation_state:
    # note that the old folder path is from the "sorted_results" file, for by-families order.
      if (dataset == "greebles_symmetric" or dataset == "greebles_asymmetric"):
        old_folder_path = fr'/home/ssd_storage/experiments/students/Noam/results_for_conference_dec21/{net_type}_net_test/vgg16/results/{dataset}_{state}/sorted_results'
      else:  
        old_folder_path = fr'/home/ssd_storage/experiments/students/Noam/results_for_conference_dec21/{net_type}_net_test/vgg16/results/{dataset}_{state}'      
      result_files = [f for f in os.listdir(old_folder_path)]
      
      for file_name in result_files:
       old_file_path = os.path.join(old_folder_path, file_name)
       layer_name = file_name[:-4]
       if layer_name not in figs_dict:
          figs_dict[layer_name] = []
       #new_file_name = layer_name + "_"+ data_type + "_"+ net_type + "_RDM" + ".html"
       #new_file_path = os.path.join(new_folder_path, new_file_name)
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
               title_text= f"RDM for {state.capitalize()} {dataset_title_name}<br>net type = {net_type}, layer = {layer_name}" ),
           # updating the y axis titles and tick labels:  
           yaxis=dict(
               tickvals = [i*32 for i in range(5)],
               ticktext = ['Family 1', 'Family 2', 'Family 3', 'Family 4', 'Family 5']),             
         )
       fig.update_xaxes(automargin=True)
       figs_dict[layer_name].append(fig)
       
       # sanity check:
       #if (layer_name == "fc7" and dataset == "bird_indiv" and net_type == "objects" and state=="upright"):
       #  fig.show()
       
       
     
for layer_name in figs_dict:
  file_path = os.path.join(results_folder_path, layer_name+ "_RDM.html")
  figures_to_html(figs_dict[layer_name], file_path)
                      

    
