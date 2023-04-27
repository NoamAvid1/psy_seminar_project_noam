import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score


def calc_graph_measurements(df: pd.DataFrame, label_col: str, value_col: str):
    """
    Given a DF with a label column {0,1} (assumes 0 is same because we usually work with distances)
    and a value column (floats)
    returns the TPR, FPR, thresholds, and AUROC
    """
    scores = [int(x) for x in df[value_col]]
    fpr, tpr, thresh = roc_curve(df[label_col], scores, pos_label=0)
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

# def calc_graph_measurments_noam(df: pd.DataFrame):
#     # Setting up data arrays for plotting:
#     data_arr = np.array(df)[:, 1:]  # array of the csv data (not including title)
#     data_vector_shape = (data_arr.shape[0] * data_arr.shape[1] - data_arr.shape[
#         0],)  # desired vector size is (160X160) - 160, Since we will delete the main diagonal
#     # create vectors for roc plotting:
#     same_pair_mat = np.ones((2, 2))
#     y_test = np.kron(np.eye((data_arr.shape[0] // 2), dtype=int),
#                      same_pair_mat)  # y_test = block diagonal matrix, each block is 2X2 matrix of "1"s.
#     # delete main diagonal (not comparing a picture to itself!):
#     y_test = y_test[~np.eye(y_test.shape[0], dtype=bool)].reshape(y_test.shape[0], -1)
#     # reshape y_test into a vector:
#     y_test = y_test.reshape(*data_vector_shape)
#     arr_norm = np.linalg.norm(data_arr)  # compute std
#     # noramlized the distances and get the "similarity" score:
#     probs = (1 - data_arr / arr_norm)
#     probs = probs[~np.eye(probs.shape[0], dtype=bool)].reshape(probs.shape[0], -1)
#     # reshape probabilities into a vector:
#     probs = probs.reshape(*data_vector_shape)
#
#     # Compute ROC curve and area the curve:
#     fpr, tpr, thresholds = roc_curve(y_test, probs)
#     roc_auc = auc(fpr, tpr)