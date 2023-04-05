import time
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import gaussian_kde
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, FeatureAgglomeration, AgglomerativeClustering, OPTICS
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
import numpy as np
import glob
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import pandas as pd
from FlowCytometryTools import FCMeasurement
import os
import warnings
import random

warnings.simplefilter('ignore')


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

selected_columns = ["Side Fluorescence Signal",
                    "Forward Scatter Signal",
                    'Side Scatter Signal']
#                    'Forward Scatter Pulse Width Signal']

SYS_FEATURES = ["PLT", "MPV", "PDW", "PCT", "IPF"]
FACS_COLS = ["FSC", "SSC", "SFL"]

datadir = "/home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/hv270/data_home"
data_dir = "/home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/hv270/data_home"

# Mostly loaders


def plot_id_exp(full_df, ID, exp):
    """
    :param full_df: Dataframe from which measurements should be taken
    :param ID: ID of the donor whose measurements we want to plot
    :param exp: type of assay
    :return: nothing, just plots all measurements contained in full_df
    """

    X = full_df.loc[(full_df.ID == ID) & (
        full_df.exp == exp)][full_df.columns[:4]]
    plt.figure()
    plt.scatter(X["Side Fluorescence Signal"],
                X["Forward Scatter Signal"], s=1)
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
                full_df.loc[(full_df.ID == ID) & (
                    full_df.exp == "wb"), "Forward Scatter Signal"], s=0.2,
                label="WB")
    plt.legend(markerscale=10)
    plt.xlabel("Side Fluorescence")
    plt.ylabel("Forward Scatter")
    plt.title(ID)


def load_fc_pheno():
    """
    :return: a dataframe summarizing the 'Gold Standard' phenotypes of ~450 donors
    """

    fc_pheno = pd.read_csv(
        "/home/hv270/data/flow_cytometry.csv", sep=";", decimal=",", skiprows=0)
    fc_pheno.set_index("ID", inplace=True)
    fc_pheno = fc_pheno[["F_ADP_AV_PP",
                         "P_ADP_AV_PP", "F_CRP_AV_PP", "P_CRP_AV_PP"]]
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
    info.loc[info["Bleed Time"].isnull(
    ), "Bleed Time"] = info.loc[info["Bleed Time"].isnull(), "Estimated Bleed Time"]
    info = info[["Sex", "Age", "Bleed Time"]]

    info["Bleed Time"] = pd.to_datetime(
        info["Bleed Time"], errors="coerce", format="%H:%M:%S")
    info["Bleed Time"] = (info["Bleed Time"] -
                          info["Bleed Time"].iloc[0]).dt.total_seconds()
    info.loc[pd.isnull(info["Bleed Time"]),
             "Bleed Time"] = info["Bleed Time"].median()
    return info


def clean_column_names(df):
    "remove brackets surrounding column names"
    to_rename = {}
    cols = df.columns
    for c in cols:
        if c[0] == "[" and c[-1] == "]":
            if c[1:-1] in cols:
                del df[c]
            else:
                to_rename[c] = c[1:-1]
    df = df.rename(columns=to_rename)
    df = df.dropna(axis=1)
    df = df
    return df


def load_sys_phenotypes(datadir, root_dir="Sysmex whole blood"):
    """
    :return: A dataframe containing IPF, PLT, PCT, MPV and PDW for all donors x assays
    """
    os.chdir(datadir)
    summaries = glob.glob("%s/**/*SAMPLE*.csv" % root_dir, recursive=True)
    print("%d candidates summary files" % len(summaries))

    dfs = [clean_column_names(pd.read_csv(f, skiprows=1)) for f in summaries]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df["Sample No."] = df["Sample No."].str.lstrip(" ")
    FB = df["Sample No."].str[-2:] == "FB"
    df.loc[FB, "Sample No."] = df.loc[FB, "Sample No."].str[:-2]
    df.rename(columns={"Sample No.": "ID"}, inplace=True)

    print("Initially %d rows, %d unique IDs" %
          (df.shape[0], df["ID"].nunique()))
    df = df.sort_values("Time", ascending=True)
    q_cols = ["PLT(10^9/L)", "IPF(%)", "MPV(fL)", "PDW(fL)", "PCT(%)"]
    cols_of_interest = q_cols + ["PLT Info."]
    for q in q_cols:
        df[q] = pd.to_numeric(df[q], errors="coerce")

    df.dropna(subset=cols_of_interest, inplace=True)
    df.drop_duplicates("ID", keep="last", inplace=True)
    print("Finally %d rows, %d unique IDs" % (df.shape[0], df["ID"].nunique()))

    df.rename(columns={"PLT(10^9/L)": "PLT",
                       "IPF(%)": "IPF",
                       "MPV(fL)": "MPV",
                       "PDW(fL)": "PDW",
                       "PCT(%)": "PCT"}, inplace=True)
    return df


def ID_from_filename(f):
    r = f.split("[")[-2][:-1]
    if r[-2:] == "FB":
        r = r[:-2]
    return r


def load_Sysmex(datadir, root_dir="Sysmex whole blood", remove_rbc=True):
    """
    :return: dataframe of all valid measurements (not only platelets)
    """
    os.chdir(datadir)
    wb = glob.glob("%s/**/*PLT-F].fcs" % root_dir, recursive=True)
    print("%d candidates Sysmex WB files" % len(wb))
    dfs = []
    IDs = {}
    for f in wb:
        meas = FCMeasurement(ID='Test Sample', datafile=f)
        df = meas.data
        ID = ID_from_filename(f)

        # Filter out those measured along with RBC
        if remove_rbc:
            df["bin_200"] = np.arange(df.shape[0]) // 200
            mean_by_200 = df.groupby("bin_200")["Forward Scatter Signal"].mean()
            try:
                cutoff = np.min(
                    np.where(mean_by_200 > np.mean(mean_by_200[:3])+30)[0])
            except ValueError as e:
                cutoff = df["bin_200"].max()
            df = df.loc[df.bin_200 < cutoff]
            del df["bin_200"]
        if ID in IDs:
            print("Two files for ID %s" % ID)
            print("\t %s" % f)
            print("\t %s" % IDs[ID])
            continue
        else:
            IDs[ID] = f
            df["ID"] = ID
            dfs.append(df)
    df = pd.concat(dfs, copy=False)
    df.dropna(axis=0, how="any", inplace=True)

    return df
  
def load_agonised_sysmex(datadir, root_dir="CBR 159", remove_rbc=True):
    """
    :return: dataframe of all valid measurements (not only platelets)
    """
    os.chdir(datadir)
    wb = glob.glob("%s/**/*PLT-F].fcs" % root_dir, recursive=True)
    print("%d candidates Sysmex WB files" % len(wb))
    dfs = []
    IDs = {}
    for f in wb:
        meas = FCMeasurement(ID='Test Sample', datafile=f)
        df = meas.data
        ID = ID_from_filename(f)

        # Filter out those measured along with RBC
        if remove_rbc:
            df["bin_200"] = np.arange(df.shape[0]) // 200
            mean_by_200 = df.groupby("bin_200")["Forward Scatter Signal"].mean()
            try:
                cutoff = np.min(
                    np.where(mean_by_200 > np.mean(mean_by_200[:3])+30)[0])
            except ValueError as e:
                cutoff = df["bin_200"].max()
            df = df.loc[df.bin_200 < cutoff]
            del df["bin_200"]
        if ID in IDs:
            print("Two files for ID %s" % ID)
            print("\t %s" % f)
            print("\t %s" % IDs[ID])
            continue
        else:
            IDs[ID] = f
            df["ID"] = ID
            dfs.append(df)
    df = pd.concat(dfs, copy=False)
    df.dropna(axis=0, how="any", inplace=True)

    return df

# Actual clustering happens here


def polygon_with_offset(offset):
    alpha = ((120+.6*offset) - (25 + .5*offset)) / (85 - 25)

    p = np.array([[22, 25, 30, 85, 140, 200, 200, 80, 50, 22],
                  [5 + .3*offset, 25 + .5*offset, max(35+.5*offset, 25+.5*offset + alpha*5), 120+.6*offset, 180+.6*offset, 220, 170, 20, 0, 5 + .3*offset]])
    #p[0] -= offset
    return p.T


def tag_platelets_of_assay(X, expected):
    #print("Using polygon")
    is_plt = {}
    plt_count = np.zeros(20)
    offsets = np.linspace(-50, 35, len(plt_count))
    for i, offset in enumerate(offsets):
        polygon = polygon_with_offset(offset)
        path = mplPath.Path(polygon)
        is_plt[i] = 1*pd.Series([(path.contains_point(pnt, radius=0.01)
                                or path.contains_point(pnt, radius=-0.01)) for pnt in X])
        plt_count[i] = is_plt[i].sum()
    diff = np.diff(plt_count)  # n of cells in the slice
    w = np.array([1, 2, 3, 2, 1])
    diff[2:-2] = np.convolve(diff, w / np.sum(w), mode="valid")

    diff2 = diff[1:]/diff[:-1]

    d_min_cells = np.argmin(diff)
    d_max_cells = np.argmax(diff)

    try:
        try:
            d_valid_diff = np.min(
                np.where(diff2[d_max_cells:] < 0.97)[0]) + d_max_cells
            d_max_diff = np.min(
                np.where(diff2[d_valid_diff:] > 0.97)[0]) + d_valid_diff
        except:
            d_max_diff = d_min_cells

        try:
            # Différence relative
            d = np.max(np.where(diff[:d_max_diff] > 1.7*diff[d_max_diff])[0])
            # d = max(d, np.max(np.where(diff[:d_max_diff] > diff[d_max_diff] + 0.1*np.max(diff))[0])) # Différence absolue
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


def tag_platelets(df, export_folder, expected_counts=None):
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
        if not os.path.exists("%s/clustering_train/%s" % (export_folder, exp)):
            os.makedirs("%s/clustering_train/%s" % (export_folder, exp))

    for ID in ids:
        n += 1

        for exp in exps:
            cond = (df.exp == exp) & (df.ID == ID)  # & cond_0
            X = df.loc[cond][['Side Fluorescence Signal',
                              'Forward Scatter Signal']].values
            expected = None
            if expected_counts is not None:
                expected = expected_counts.loc[ID]
            PLT = tag_platelets_of_assay(X.copy(), expected)
            df.loc[cond, "PLT"] = PLT.values
            # if exp == "wb":
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
    counts = df.loc[df.PLT > 0].groupby(
        ["ID", "exp"])[["Side Fluorescence Signal"]].count().reset_index()
    counts = counts.rename(
        columns={"Side Fluorescence Signal": "PLT_count_filter"})
    comp = pd.merge(sys_phen, counts, left_on=[
                    "ID", "exp"], right_on=["ID", "exp"], how="inner")
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
            rel_err = np.abs(residuals / (2 * y - residuals)
                             )  # = (y-pred) / (y+ pred)
            wrong_IDs += comp.loc[cond].iloc[np.where(
                rel_err > 0.1)]["ID"].unique().tolist()
            plt.tight_layout()

        for ID in wrong_IDs:
            print("ID = %s, exp = %s" % (ID, exp))
            print(((df.ID == ID) & (df.exp == exp)).sum())
            if ((comp.exp == exp) & (comp.ID == ID)).sum() > 1:
                print(comp.loc[(comp.exp == exp) & (comp.ID == ID)])
            plt.figure()
            plt.scatter(df.loc[(df.ID == ID) & (df.exp == exp), "Side Fluorescence Signal"],
                        df.loc[(df.ID == ID) & (df.exp == exp),
                               "Forward Scatter Signal"],
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


def get_INTERVAL_phenotypes():
    INTERVAL_path = "/home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/hippolyte_only"
    print("Loading INTERVAL phenotypes")
    pheno = pd.read_csv("%s/interval_phenotypes/INTERVALdata_14MAY2020.csv" %
                        INTERVAL_path, sep=",", encoding="ISO-8859-1")
    pheno = pheno.rename(columns={"agePulse": "Age", "sexPulse": "Sex"})
    pheno = pheno.loc[~pheno.Age.isnull() & ~pheno.Sex.isnull()]
    print(pheno.shape)
    pheno = pheno.loc[~pheno.interval.isnull()]
    print(pheno.shape)
    pheno = pheno.loc[pheno.ethnicPulse == "Eng/W/Scot/NI/Brit"]
    print(pheno.shape)

    meta = pd.read_csv("%s/interval_HAAS/meta_table.tsv" %
                       INTERVAL_path, sep="\t")
    print(meta.shape)
    meta = meta.loc[meta.SampleType == "NHSBT"]
    print(meta.shape)

    # Select among various measurements
    pheno_bl = pheno.rename(columns={'PLT_10_9_L_bl': "PLT",
                                                      'MPV_fL_bl': "MPV",
                                                      'PCT_PCT_bl': "PCT",
                                                      'PDW_fL_bl': "PDW",
                                                      "IPF_bl": "IPF"}).copy()
    pheno_bl.dropna(subset=SYS_FEATURES, how="any", inplace=True)
    pheno_bl["in_cohort"] = "BL"

    int_phen = pd.merge(meta.loc[~meta.EpiCovId_bl.isnull()], pheno_bl,
                        left_on="EpiCovId_bl", right_on="identifier", how="inner",
                        validate="one_to_one")
    print(int_phen.shape)

    pheno_24 = pheno.loc[~pheno.identifier.isin(
        int_phen.identifier.tolist())].copy()
    pheno_24.rename(columns={'PLT_10_9_L_24m': "PLT",
                             'MPV_fL_24m': "MPV",
                             'PCT_PCT_24m': "PCT",
                             'PDW_fL_24m': "PDW",
                             "IPF_24m": "IPF"}, inplace=True)
    pheno_24.dropna(subset=["PLT", "MPV", "PDW", "PCT",
                    "IPF"], how="any", inplace=True)
    pheno_24["in_cohort"] = "24"

    to_add = pd.merge(meta.loc[~meta["EpiCovId_24m"].isnull()],
                      pheno_24,
                      left_on="EpiCovId_24m", right_on="identifier",
                      validate="one_to_one")
    int_phen = pd.concat([int_phen, to_add], copy=False, axis=0)
    print(int_phen.shape)

    pheno_48 = pheno.loc[~pheno.identifier.isin(
        int_phen.identifier.tolist())].copy()
    pheno_48.rename(columns={'PLT_10_9_L_48m': "PLT",
                             'MPV_fL_48m': "MPV",
                             'PCT_PCT_48m': "PCT",
                             'PDW_fL_48m': "PDW",
                             "IPF_48m": "IPF"}, inplace=True)
    pheno_48.dropna(subset=SYS_FEATURES, how="any", inplace=True)
    pheno_48["in_cohort"] = "48"

    to_add = pd.merge(meta.loc[~meta["EpiCovId_48m"].isnull()],
                      pheno_48,
                      left_on="EpiCovId_48m", right_on="identifier",
                      validate="one_to_one")
    int_phen = pd.concat([int_phen, to_add], copy=False, axis=0)
    print(int_phen.shape)

    assert int_phen.SampleNo.nunique() == int_phen.shape[0]
    print("Checked that each Sample has one row max")
    
    print("Removing samples with less than 10 measurements per day")
    int_phen["DateTime"] = pd.to_datetime(int_phen["DateTime"])
    int_phen["day"] = int_phen["DateTime"].dt.date
    int_phen["machine"] = 1+1*(int_phen.Instrument == "XN-10^11041")
    int_phen["weekday"] = int_phen["DateTime"].dt.weekday
    int_phen["yearday"] = int_phen.DateTime.dt.day_of_year
    int_phen["hours"] = int_phen.DateTime.dt.hour
    
    exp_by_day_machine = int_phen.groupby(["machine","day"]).SampleNo.nunique()
    good_day_machine = exp_by_day_machine.loc[exp_by_day_machine >= 10]
    int_phen = int_phen.set_index(["machine","day"])
    int_phen = int_phen.loc[int_phen.index.isin(good_day_machine.index.tolist())].reset_index()
    print(int_phen.shape)
    
    
    return int_phen, meta


def get_ages_sex():

    pheno_meta_df = pd.read_csv(
        "%s/metadata_PF/PLATELET_FUNCTION_METADATA_KD_11062019.txt" % data_dir, sep="\t")
    sys_info = pheno_meta_df.set_index("SAMPLE_ID")[["SEX", "AGE"]]
    sys_info["SEX"] = 1+1*(sys_info["SEX"] == "F")  # 1 if man, 2 if woman

    int_phen, _ = get_INTERVAL_phenotypes()
    int_info = int_phen.set_index("SampleNo")[["Sex", "Age"]].rename(
        columns={"Age": "AGE", "Sex": "SEX"})

    return sys_info, int_info
