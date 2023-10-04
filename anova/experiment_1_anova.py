import pandas as pd
import pingouin as pg
import os
from statsmodels.stats.anova import AnovaRM

experiment_folder = "..\\seminar_dists_results\\experiment_1_2\\batches_aucs"

dependent_var = "auc"
between_factor = "dcnn_domain"
within_factors = ['img_domain', 'orientation']

if os.path.exists(r"..\seminar_dists_results\experiment_1_2\batches_aucs\all.csv"):
    df = pd.read_csv(r"..\seminar_dists_results\experiment_1_2\batches_aucs\all.csv")
else:
    df = pd.DataFrame()
    for file_name in os.listdir(experiment_folder):
        # columns are: dcnn_domain, img_domain, orientation, batch (1-30), auc
        new_df_path  = f"{experiment_folder}/{file_name}"
        new_df = pd.read_csv(new_df_path)
        df = pd.concat([df, new_df], ignore_index=True)

    # add subject ids:
    df = df[[dependent_var, between_factor, *within_factors, 'id']]
    df.to_csv(f"{experiment_folder}/all.csv")

