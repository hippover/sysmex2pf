import warnings
import random

warnings.simplefilter('ignore')
import os
from FlowCytometryTools import FCMeasurement
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import glob

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, FeatureAgglomeration, AgglomerativeClustering, OPTICS
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from scipy.stats import gaussian_kde
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from tqdm import tqdm

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
                  (df_["Side Fluorescence Signal"] <= 220) & \
                  (df_["Forward Scatter Signal"] <= 200)

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

def polygon_with_offset(offset):
    alpha = ((120+.6*offset) - (25+ .5*offset)) / (85 - 25)
    
    p = np.array([[22,25,30,85 ,140,200,200,80,50,22],
                  [5 + .3*offset ,25+ .5*offset,max(35+.5*offset,25+.5*offset + alpha*5),120+.6*offset,180+.6*offset,220,170,20,0,5+ .3*offset]])
    #p[0] -= offset
    return p.T

def tag_platelets_of_assay(X, expected):
    #print("Using polygon")
    is_plt = {}
    plt_count = np.zeros(30)
    offsets = np.linspace(-50,35,len(plt_count))
    for i, offset in enumerate(offsets):
        polygon = polygon_with_offset(offset)
        path = mplPath.Path(polygon)
        is_plt[i] =  1*pd.Series([(path.contains_point(pnt,radius=0.01) or path.contains_point(pnt,radius=-0.01)) for pnt in X])
        plt_count[i] = is_plt[i].sum()
    diff = np.diff(plt_count) # n of cells in the slice
    w = np.array([1,2,3,2,1])
    diff[2:-2] = np.convolve(diff, w / np.sum(w),mode="valid")
    
    diff2 = diff[1:]/diff[:-1]
    
    d_min_cells = np.argmin(diff)
    d_max_cells = np.argmax(diff)
    
    try:
        try:
            d_valid_diff = np.min(np.where(diff2[d_max_cells:] < 0.97)[0]) + d_max_cells
            d_max_diff = np.min(np.where(diff2[d_valid_diff:] > 0.97)[0]) + d_valid_diff
        except:
            d_max_diff = d_min_cells
            
        try:
            d = np.max(np.where(diff[:d_max_diff] > 1.7*diff[d_max_diff])[0]) # Différence relative
            #d = max(d, np.max(np.where(diff[:d_max_diff] > diff[d_max_diff] + 0.1*np.max(diff))[0])) # Différence absolue
        except:
            d = d_max_diff
        #print(d_min_cells,d_max_diff, d)
        

    except:
    
        plt.figure()
        plt.title("Diff")
        plt.plot(diff)
        plt.figure()
        plt.title("Diff2")
        plt.plot(diff2)
        raise
    """
    plt.figure()
    plt.scatter(X[:,0],X[:,1],s=.1)
    plt.plot(polygon_with_offset(offsets[d_min_cells])[:,0],
             polygon_with_offset(offsets[d_min_cells])[:,1], label="Min cells")
    plt.plot(polygon_with_offset(offsets[d_max_diff])[:,0],
             polygon_with_offset(offsets[d_max_diff])[:,1], label="rebounce")
    plt.plot(polygon_with_offset(offsets[d])[:,0],
             polygon_with_offset(offsets[d])[:,1], label="chosen = %d" % d)
    plt.legend()
    """
    return is_plt[int(d)]
    


def tag_platelets(df, export_folder, expected_counts = None):
    """
    This is where platelets are tagged
    A few cells are discarded by hand using simple gating, partly to reduce
    unnecessary computing of clusters.
    The rest is tagged using DBSCAN
    :param df: all Sysmex measurements
    :return: A dataframe containing all measurements and a column PLT
    equal to 1 if the cell is a platelet
    """

    df["PLT"] = 0
    n = 0
    ids = df.ID.unique().tolist()
    ids = np.random.permutation(ids)

    exps = df.exp.unique()
    for exp in exps:
        if not os.path.exists("%s/clustering_train/%s" % (export_folder,exp)):
            os.makedirs("%s/clustering_train/%s" % (export_folder,exp))

    for ID in ids:
        n += 1
        
        for exp in exps:
            cond = (df.exp == exp) & (df.ID == ID)  #& cond_0
            X = df.loc[cond][['Side Fluorescence Signal', 'Forward Scatter Signal']].values
            expected = None
            if expected_counts is not None:
                expected = expected_counts.loc[ID]
            PLT = tag_platelets_of_assay(X.copy(), expected)
            df.loc[cond, "PLT"] = PLT.values
            #if exp == "wb":
            #    fig = plt.figure()
            #    ax = fig.add_subplot(111)
            #    ax.scatter(df.loc[cond,"Side Fluorescence Signal"],
            #        df.loc[cond,"Forward Scatter Signal"],
            #        c=df.loc[cond,"PLT"],s=1)
            #    ax.set_title("%s %s" % (ID,exp))
            #    plt.savefig("%s/clustering_train/%s/%s.png" % (export_folder,exp,ID))

    #print("%.2f  of cells are platelets" % (df["PLT"].mean()))

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
