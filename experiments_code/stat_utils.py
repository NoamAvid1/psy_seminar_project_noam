import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

NUM_BATCHES = 30


def calc_graph_measurements(df: pd.DataFrame, label_col: str, value_col: str):
    """
    Given a DF with a label column {0,1} (assumes 0 is same because we usually work with distances)
    and a value column (floats)
    returns the TPR, FPR, thresholds, and AUROC
    """
    # scores for L2 distances?
    # scores = [int(x) for x in df[value_col]]

    fpr, tpr, thresh = roc_curve(df[label_col], df[value_col], pos_label=0)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, thresh, roc_auc


def remove_rdm_redundancies(rdm:pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records from the RDM (choose only upper triangle)
    Remove distances from an item to itself (remove the main diagonal)
    """
    temp = rdm.copy()
    lt = np.tril(np.ones(rdm.shape), -1).astype(np.bool)

    temp = temp.where(lt == False, np.nan)
    np.fill_diagonal(temp.values, np.nan)
    return temp


def rdm_to_dist_list(rdm: pd.DataFrame, img_to_class_path: str = "") -> pd.DataFrame:
    """
    Given a full RDM, remove redundancies and return it in a format of a distance list
    """
    rdm = rdm.iloc[:, 1:]  # remove column of images names
    rdm = remove_rdm_redundancies(rdm)
    # csv that maps every image to its class:
    img_to_class = pd.read_csv(img_to_class_path, index_col=0)
    # index and columns are of format <class_name>:<image_name>"
    rdm.index = [str(img_to_class.loc[row][0]) + ':' + str(row) for row in img_to_class.index]
    rdm.columns = [str(img_to_class.loc[row][0]) + ':' + str(row) for row in img_to_class.index]
    stacked = rdm.stack()
    no_diag = pd.DataFrame(stacked.dropna()).rename(columns={0: 'cos'})
    same = []
    for idx in no_diag.index:
        same.append(idx[0].split(':')[0] == idx[1].split(':')[0])
    no_diag['same'] = same
    return no_diag


def is_same_class(pair):
    """
    Takes a pairs list entry in the format "(class1/image1_name, class2/image2_name)"
    and checks if both images belong to the same class
    """
    return eval(pair)[0].split('/')[0] == eval(pair)[1].split('/')[0]


def pairs_list_to_dist_list(df: pd.DataFrame, dataset_type):
    """
    Given df of pair and dist, add "same" column and rename to "cos" column
    """
    df = df[(df['type'] == dataset_type)].copy()
    assert (len(df) > 0)
    df['same'] = df.iloc[:, 0].apply(is_same_class)
    df.rename({df.columns[1]: 'cos'}, axis=1, inplace=True)
    return df


def get_df_auc_mean_std(df: pd.DataFrame, dcnn_domain, img_domain, orientation, experiment_name):
    same_df = df[df['same']]
    same_df = same_df.sample(frac=1)
    diff_df = df[~df['same']]
    diff_df = diff_df.sample(frac=1)
    assert(len(same_df) == len(diff_df))
    aucs = []
    fprs = []
    tprs = []
    batch_size = len(diff_df) // NUM_BATCHES
    for i in range(NUM_BATCHES):
        test_df = pd.concat([same_df.iloc[i * batch_size: (i + 1) * batch_size, ],
                            diff_df.iloc[i * batch_size: (i + 1) * batch_size, ]])
        fpr, tpr, thresh, roc_auc = calc_graph_measurements(test_df, 'same', 'cos')
        aucs.append(roc_auc)
        fprs.append(fpr)
        tprs.append(tpr)

    # fpr_mean = np.mean(np.array(fprs), axis=0)
    # tpr_mean = np.mean(np.array(tprs), axis=0)
    batch_names = [f"{dcnn_domain}_{i}" for i in range(1, NUM_BATCHES + 1)]
    auc_mean, auc_std = np.mean(aucs), np.std(aucs)
    res_df = pd.DataFrame({"id": batch_names,"auc": aucs})
    res_df["dcnn_domain"] = dcnn_domain
    res_df["img_domain"] = img_domain
    res_df["orientation"] = orientation

    res_df.to_csv(f"../seminar_dists_results/{experiment_name}/batches_aucs/{dcnn_domain}_{img_domain}_{orientation}.csv")
    return auc_mean, auc_std


