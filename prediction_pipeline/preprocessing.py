import warnings
import random

warnings.simplefilter('ignore')
import os
from FlowCytometryTools import FCMeasurement
import pandas as pd
import matplotlib.pyplot as plt
import glob

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, FeatureAgglomeration, AgglomerativeClustering, OPTICS
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from scipy.stats import gaussian_kde
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from tqdm.notebook import tqdm

import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

selected_columns = ["Side Fluorescence Signal",
                    "Forward Scatter Signal",
                    'Side Scatter Signal']
#                    'Forward Scatter Pulse Width Signal']

datadir = "/home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/hv270/data_home"

# Mostly loaders

def build_df(files, exp):
    """
    :param files: list of FACS files
    :param exp: type of assay (rest, adp, ...)
    :return: dataframe containing all valid measurements contained in the files list
    """

    columns = ["Side Fluorescence Signal",
               "Forward Scatter Signal",
               'Side Scatter Signal']
#               'Forward Scatter Pulse Width Signal']

    users = [f.split('][')[-2].split("_")[0] for f in files]
    for i in range(len(users)):
        if users[i][-2:] == "FB":
            users[i] = users[i][:-2]

    dfs = []
    appended_IDs = []
    for k in range(len(users)):
        ID = users[k]
        if ID not in appended_IDs:
            meas = FCMeasurement(ID='Test Sample', datafile=files[k]).data[columns]
            meas["ID"] = ID
            dfs.append(meas.copy())
            appended_IDs.append(users[k])
        else:
            print("Duplicate file for ID %s : %s" % (ID,files[k]))

    df_ = pd.concat(dfs, copy=False)
    df_.reset_index(inplace=True)

    del df_["index"]
    df_["exp"] = exp
    print(exp, len(df_.ID.unique().tolist()))

    select_cond = (df_["Side Fluorescence Signal"] > 0.1) & \
                  (df_["Side Fluorescence Signal"] <= 220)

    return df_.loc[select_cond]


def load_Sysmex(datadir):
    """
    :return: dataframe of all valid measurements (not only platelets)
    """
    os.chdir(datadir)
    wb = glob.glob("Sysmex whole blood/**/*PLT-F].fcs", recursive=True)
    print("%d candidates Sysmex WB files" % len(wb))

    files_exp = [[wb, "wb"]]

    return pd.concat([build_df(f[0], f[1]) for f in files_exp], copy=False)


def plot_id_exp(full_df, ID, exp):
    """
    :param full_df: Dataframe from which measurements should be taken
    :param ID: ID of the donor whose measurements we want to plot
    :param exp: type of assay
    :return: nothing, just plots all measurements contained in full_df
    """

    X = full_df.loc[(full_df.ID == ID) & (full_df.exp == exp)][full_df.columns[:4]]
    plt.figure()
    plt.scatter(X["Side Fluorescence Signal"], X["Forward Scatter Signal"], s=1)
    plt.xlabel("Side Fluorescence Signal")
    plt.ylabel("Forward Scatter Signal")
    plt.title(ID)


def example_plot(full_df, ID):
    """
    :param full_df: dataframe containing measurements
    :param ID: ID of the donors
    :return: nothing, it just plots ADP, CRP and rest assays
    """
    plt.scatter(full_df.loc[(full_df.ID == ID) & (full_df.exp == "wb"), "Side Fluorescence Signal"],
                full_df.loc[(full_df.ID == ID) & (full_df.exp == "wb"), "Forward Scatter Signal"], s=0.2,
                label="WB")
    plt.legend(markerscale=10)
    plt.xlabel("Side Fluorescence")
    plt.ylabel("Forward Scatter")
    plt.title(ID)


def load_fc_pheno():
    """
    :return: a dataframe summarizing the 'Gold Standard' phenotypes of ~450 donors
    """

    fc_pheno = pd.read_csv("/home/hv270/data/flow_cytometry.csv", sep=";", decimal=",", skiprows=0)
    fc_pheno.set_index("ID", inplace=True)
    fc_pheno = fc_pheno[["F_ADP_AV_PP", "P_ADP_AV_PP", "F_CRP_AV_PP", "P_CRP_AV_PP"]]
    fc_pheno[["F_ADP_AV_PP", "P_ADP_AV_PP", "F_CRP_AV_PP", "P_CRP_AV_PP"]] = np.log(
        fc_pheno[["F_ADP_AV_PP", "P_ADP_AV_PP", "F_CRP_AV_PP", "P_CRP_AV_PP"]] / 100) - np.log(
        1 - fc_pheno[["F_ADP_AV_PP", "P_ADP_AV_PP", "F_CRP_AV_PP", "P_CRP_AV_PP"]] / 100)
    return fc_pheno


def load_info():
    """
    :return: a dataframe containing information about donors (age, sex, bleeding time, etc...)
    """

    info = pd.read_excel("/home/hv270/data/CBR_159_FACS_LMD/PFC4 Healthy Controls_159 GPVI_Sample Record_22012019.xlsx",
                         sep=";", decimal=",", skiprows=0)
    info.Sex = 1 * (info.Sex == "F")
    info.set_index("CFT 4R ID", inplace=True)
    info.loc[info["Bleed Time"].isnull(), "Bleed Time"] = info.loc[info["Bleed Time"].isnull(), "Estimated Bleed Time"]
    info = info[["Sex", "Age", "Bleed Time"]]

    info["Bleed Time"] = pd.to_datetime(info["Bleed Time"], errors="coerce", format="%H:%M:%S")
    info["Bleed Time"] = (info["Bleed Time"] - info["Bleed Time"].iloc[0]).dt.total_seconds()
    info.loc[pd.isnull(info["Bleed Time"]), "Bleed Time"] = info["Bleed Time"].median()
    return info


def load_sys_phenotypes(datadir):
    """
    :return: A dataframe containing IPF, PLT, PCT, MPV and PDW for all donors x assays
    """
    os.chdir(datadir)
    summaries = glob.glob("Sysmex whole blood/**/*SAMPLE*.csv",recursive=True)
    print("%d candidates summary files" % len(summaries))
    IDs = list()
    exps = list()
    PLTs = list()
    IPFs = list()
    is_PLT_Fs = list()
    MPVs = list()
    PDWs = list()
    PCTs = list()
    dates = list()

    n_errors = 0
    for f in summaries:
        s = pd.read_csv(f, skiprows=1)
        ID = [s["Sample No."].iloc[i][:-2].lstrip(" ") for i in range(s.shape[0])]
        exp = ["wb" for i in range(s.shape[0])]
        date = [f.split("/")[2].split("_")[0] for k in range(s.shape[0])]
        
        try:
            PLT = s["PLT(10^9/L)"].tolist()
            IPF = s["IPF(%)"].tolist()
            MPV = s["MPV(fL)"].tolist()
            PDW = s["PDW(fL)"].tolist()
            PCT = s["PCT(%)"].tolist()
            is_PLT_F = [1 * (source == "PLT-F") for source in s["PLT Info."]]
        except BaseException as error:
            print(f)
            print(s.columns)
            print(error)
            n_errors += 1
            continue

        IDs += ID
        exps += exp
        PLTs += PLT
        IPFs += IPF
        is_PLT_Fs += is_PLT_F
        MPVs += MPV
        PDWs += PDW
        PCTs += PCT
        dates += date

    plt_df = pd.DataFrame.from_dict(data={"ID": IDs, "exp": exps, "PLT_count": PLTs,
                                          "IPF": IPFs, "is_PLT_F": is_PLT_Fs,
                                          "MPV": MPVs, "PDW": PDWs, "PCT": PCTs, "date": dates
                                          }, orient="columns")

    for c in ["IPF", "MPV", "PDW", "PCT", "PLT_count"]:
        if plt_df[c].dtype not in [float, int]:
            print("Missing values of %s : %.2f" % (c, (plt_df[c] == "----").mean()))
            plt_df.loc[plt_df[c] == "----", c] = plt_df.loc[plt_df[c] != "----", c].median()
            plt_df[c] = plt_df[c].astype(float)
    print("Reading errors : %d" % n_errors)
    plt_df = plt_df.sort_values("is_PLT_F", ascending=False)
    plt_df = plt_df.drop_duplicates(["ID", "exp"], keep="first")

    return plt_df

# Actual clustering happens here

def get_clusters(X_t):
    """
    clusters cells using DBSCAN
    :param X_t: n x 2 array containing ['Side Fluorescence Signal', 'Forward Scatter Signal']
    :return: labels
    """
    th_0 = 110
    th_1 = 95
    # To keep density roughly uniform, I contract the upper-tail of the platelet cloud
    # X_t[np.where(X_t[:, 0] > th), 0] = th + 0.1 * (X_t[np.where(X_t[:, 0] > th), 0] - th)
    X_t[:, 0] = np.clip(X_t[:, 0], a_min=0, a_max=th_0) + 0.1 * np.clip(X_t[:, 0] - th_0, a_min=0, a_max=np.inf)
    X_t[:, 1] = np.clip(X_t[:, 1], a_min=0, a_max=th_1) + 0.2 * np.clip(X_t[:, 1] - th_1, a_min=0, a_max=np.inf)
    X_t[:, 0] = np.clip(X_t[:, 0], a_min=0, a_max=40) + 0.8 * np.clip(X_t[:, 0] - 40, a_min=0, a_max=np.inf)
    X_t[:, 0] = 30 + np.clip(X_t[:, 0] - 30, a_min=0, a_max=np.inf) - 1.5 * np.clip(30 - X_t[:,0], a_min=0, a_max=30)

    # I also extend the bottom tail
    th_2 = 30
    th_3 = 20
    #X_t[2*X_t[:,0] + X_t[:,1] < 70] *= 0.7
    #X_t[:, 0] = np.clip(X_t[:, 0], a_min=th_2, a_max=np.inf) + 1.5 * np.clip(X_t[:, 0] - th_2, a_min=-np.inf, a_max=0)
    #X_t[:, 1] = np.clip(X_t[:, 1], a_min=th_3, a_max=np.inf) + 1.5 * np.clip(X_t[:, 1] - th_3, a_min=-np.inf, a_max=0)


    ms = max(X_t.shape[0] / 800, 10)
    #eps = 4 + 5.5 * np.exp(-X_t.shape[0] / 3e3)
    eps = 3.3

    # According to documentation, DBSCAN is more memoy-efficient
    # when the neighboring graph is precomputed.
    nn = NearestNeighbors(radius=eps, n_jobs=-1)
    nn.fit(X_t)
    neighbors = nn.radius_neighbors_graph(X_t, mode='distance')
    db = DBSCAN(eps=eps, n_jobs=-1, min_samples=ms, metric='precomputed').fit(neighbors)
    
    """
    plt.figure()

    for label in np.unique(db.labels_):
        plt.scatter(X_t[db.labels_ == label][:,0],
                    X_t[db.labels_ == label][:,1],s=0.5)
    """
    return db.labels_

def refine_clustering(X,labels,selected_cluster):
    """
    Sometimes, cells which are a bit above the platelets cloud are included. We get rid of them.
    :param X:
    :param labels:
    :param selected_cluster:
    :return:
    """

    Y = X[labels==selected_cluster]
    pca = PCA().fit(Y)
    if pca.components_[1,0] > 0:
        pca.components_[1,:] = - pca.components_[1,:]

    # This is a bit cheating, but I'm rorating a bit the PCA matrix to better cut out the outliers
    alpha=-0.035
    M = np.array([[np.cos(alpha),np.sin(alpha)],[-np.sin(alpha),np.cos(alpha)]])
    pca.components_ = M @ pca.components_
    X_t = pca.transform(Y)

    # Detect the discontinuity, if there is one
    m = np.median(X_t[:,1]) - 4*np.std(X_t[:,1])
    M = np.median(X_t[:,1]) + 4*np.std(X_t[:,1])
    n_bins = 35
    hist,edges = np.histogram(X_t[:,1],bins=np.linspace(m,M,n_bins),density=False)
    summit = np.argmax(hist)
    first_0 = n_bins-2

    min_hist = 3 # If a slice has less than min_hist cells, then we consider this slice to be a gap and the end of platelets.
    for k in range(summit,n_bins-2):
        if hist[k] < min_hist:
            first_0 = k
            break
        elif hist[k-1] + hist[k] < 2.5*min_hist:
            first_0 = k
            break
    first_0_index = first_0
    first_0 = edges[first_0_index]
    labels[labels == selected_cluster] = selected_cluster*(X_t[:,1] < first_0) + 1*(X_t[:,1] >= first_0)
    return labels

def tag_platelets_of_assay(X):
    labels = get_clusters(X.copy())

    valid_labels = labels[np.where(X[:, 1] < 150)]  # Used to determine good cluster
    labels_unique = np.unique(valid_labels, return_counts=True)
    sel_c_index = np.argmax(labels_unique[1])
    selected_cluster = labels_unique[0][sel_c_index]
    # print(len(valid_labels),selected_cluster,sel_c_index)

    labels = refine_clustering(X,labels,selected_cluster)
    # We check if the cluster doesn't go left until the edge
    #if np.min(X[:,0]) < np.min(X[labels == selected_cluster,0]) * 1.01:
    #

    return 1 * (labels == selected_cluster)

def tag_platelets(df, export_folder):
    """
    This is where platelets are tagged
    A few cells are discarded by hand using simple gating, partly to reduce
    unnecessary computing of clusters.
    The rest is tagged using DBSCAN
    :param df: all Sysmex measurements
    :return: A dataframe containing all measurements and a column PLT
    equal to 1 if the cell is a platelet
    """

    print("Tag platelets")

    df["PLT"] = 0
    n = 0
    ids = df.ID.unique().tolist()
    ids = np.random.permutation(ids)

    exps = df.exp.unique()
    for exp in exps:
        if not os.path.exists("%s/clustering_train/%s" % (export_folder,exp)):
            os.makedirs("%s/clustering_train/%s" % (export_folder,exp))

    for ID in tqdm(ids):
        n += 1

        '''cond_0 = (df["Side Fluorescence Signal"] > 17.5) \
                                    & (df.ID == ID) \
                                    & (df["Forward Scatter Pulse Width Signal"] > 0.1) \
                                    & (df["Side Fluorescence Signal"] <= 220) \
                                    & (df["Forward Scatter Signal"] < 190)'''

        cond_0 = (df["Side Fluorescence Signal"] > 17.5) \
            & (df.ID == ID) \
            & (df["Side Fluorescence Signal"] <= 220) \
            & (df["Forward Scatter Signal"] < 190) \
            & (df["Forward Scatter Signal"] < df ["Side Fluorescence Signal"] + 70)

        for exp in exps:
            cond = (df.exp == exp) & cond_0
            X = df.loc[cond][['Side Fluorescence Signal', 'Forward Scatter Signal']].values

            df.loc[cond, "PLT"] = tag_platelets_of_assay(X)

            if exp == "wb":

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(df.loc[cond,"Side Fluorescence Signal"],
                    df.loc[cond,"Forward Scatter Signal"],
                    c=df.loc[cond,"PLT"],s=1)
                ax.set_title("%s %s" % (ID,exp))
                plt.savefig("%s/clustering_train/%s/%s.png" % (export_folder,exp,ID))

    print("%.2f  of cells are platelets" % (df["PLT"].mean()))

    return df


def filter_with_count(df, sys_phen):
    """
    Plots a comparison of platelet count provided by Sysmex to
    the one found by DBSCAN clustering
    :param df: dataframe of tagged cells
    :param sys_phen: dataframe containing PLT for all assays
    :return: nothing
    """
    counts = df.loc[df.PLT > 0].groupby(["ID", "exp"])[["Side Fluorescence Signal"]].count().reset_index()
    counts = counts.rename(columns={"Side Fluorescence Signal": "PLT_count_filter"})
    comp = pd.merge(sys_phen, counts, left_on=["ID", "exp"], right_on=["ID", "exp"], how="inner")
    #n_exp = comp.groupby("ID")["exp"].count()
    #comp = comp.loc[comp.ID.isin(n_exp.loc[n_exp == 4].index.tolist())]
    fig = plt.figure(figsize=(8, 8))
    i = 1
    exps = comp.exp.unique().tolist()
    print(exps)
    for exp in exps:

        ax = fig.add_subplot()
        scores = [0, 0]
        wrong_IDs = []
        for p in [1]:
            cond = (comp.exp == exp)  # & (comp.is_PLT_F == p)
            if cond.sum() == 0:
                continue
            x = comp.loc[cond, "PLT_count"]
            y = comp.loc[cond, "PLT_count_filter"]
            ids = comp.loc[cond, "ID"]

            legend = "PLT-F count"
            if p == 0:
                legend = "PLT-O count"
            ax.scatter(x, y, label=legend, s=5)
            ax.set_xlabel("Sysmex PLT count (10^9/L)")
            ax.set_ylabel("# platelets found")
            lm = LinearRegression(fit_intercept=True, normalize=True)
            lm.fit(np.reshape(np.array(x), (-1, 1)), y)
            scores[p] = lm.score(np.reshape(np.array(x), (-1, 1)), y)
            residuals = y - lm.predict(np.reshape(np.array(x), (-1, 1)))
            rel_err = np.abs(residuals / (2 * y - residuals))  # = (y-pred) / (y+ pred)
            wrong_IDs += comp.loc[cond].iloc[np.where(rel_err > 0.1)]["ID"].unique().tolist()
            plt.tight_layout()

        for ID in wrong_IDs:
            print("ID = %s, exp = %s" % (ID, exp))
            print(((df.ID == ID) & (df.exp == exp)).sum())
            if ((comp.exp == exp) & (comp.ID == ID)).sum() > 1:
                print(comp.loc[(comp.exp == exp) & (comp.ID == ID)])
            plt.figure()
            plt.scatter(df.loc[(df.ID == ID) & (df.exp == exp), "Side Fluorescence Signal"],
                        df.loc[(df.ID == ID) & (df.exp == exp), "Forward Scatter Signal"],
                        c=df.loc[(df.ID == ID) & (df.exp == exp), "PLT"], s=5, cmap=plt.cm.RdYlGn)
            plt.title("%s - %s | Sysmex platelet counts : %.2f | Custering count : %.2f" %
                      (ID, exp,
                       (comp.loc[cond & (comp.ID == ID), "PLT_count"].mean() - comp.loc[cond, "PLT_count"].min()) / (
                               comp.loc[cond, "PLT_count"].max() - comp.loc[cond, "PLT_count"].min()),
                       (comp.loc[cond & (comp.ID == ID), "PLT_count_filter"].mean() - comp.loc[
                           cond, "PLT_count_filter"].min()) / (
                               comp.loc[cond, "PLT_count_filter"].max() - comp.loc[cond, "PLT_count_filter"].min())))

        print("Spotted %d samples from %s" % (len(wrong_IDs), exp))
        # df = df.loc[~(df.ID.isin(wrong_IDs) & (df.exp == exp))]
        ax.set_title("%s - $R^2 = %.2f$" % (exp.upper(), scores[1]))
        # ax.legend()
        i += 1
    plt.tight_layout()
