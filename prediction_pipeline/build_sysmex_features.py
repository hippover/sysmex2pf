import pickle
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, RobustScaler
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

selected_columns = ["Side Fluorescence Signal",
                    "Forward Scatter Signal",
                    'Side Scatter Signal']
#                    'Forward Scatter Pulse Width Signal']

def initials(s):
    r = "".join([w[0] for w in s.split(" ")])
    if r == "FSS":
        r = "FSC"
    elif r == "SFS":
        r = "SFL"
    return r

def mode(x):
    """
    :param x: array
    :return: mode of x
    """
    q,b = np.histogram(x,bins=15)
    return 0.5*(b[np.argmax(q)] + b[np.argmax(q) + 1])

def entropy(x):
    p,_ = np.histogram(x,bins=100,density=False)
    p = p / np.sum(p)
    return -np.nansum(p*np.log(p))

def kernel_estimates(df,steps):
    """
    :param df: Sysmex measurements (PC 1 and 2)
    :return: kernel densities in
    """
    hor_steps, vert_steps = steps
    kernel = gaussian_kde(df[[0,1]].transpose())# ,bw_method=0.31
    #X, Y = np.mgrid[-6:6:complex(0,hor_steps), -2.5:2:complex(0,vert_steps)]
    #X, Y = np.mgrid[-4:7:complex(0, hor_steps), -2:2:complex(0, vert_steps)]

    X = np.quantile(df[0],np.linspace(0.01,0.99,hor_steps))
    Y = np.quantile(df[1],np.linspace(0.01,0.99,vert_steps))
    X,Y = np.meshgrid(X,Y)

    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)
    return pd.Series(data=np.reshape(Z,(-1)),index=["KDE_%d"% f for f in np.arange(hor_steps*vert_steps)])

def get_features_per_ID(X,mpv,plt_count,pdw,pct,hor_steps,vert_steps):

    g = X.groupby("ID")

    high_10_0 = g[0].quantile(0.9)
    high_10_1 = g[1].quantile(0.9)
    low_10_0 = g[0].quantile(0.1)
    low_10_1 = g[1].quantile(0.1)
    high_20_0 = g[0].quantile(0.95)
    high_20_1 = g[1].quantile(0.95)
    low_20_0 = g[0].quantile(0.05)
    low_20_1 = g[1].quantile(0.05)
    high_50_0 = g[0].quantile(0.98)
    high_50_1 = g[1].quantile(0.98)
    low_50_0 = g[0].quantile(0.02)
    low_50_1 = g[1].quantile(0.02)
    std_0 = g[0].std()
    std_1 = g[1].std()
    mean_0 = g[0].mean()
    mean_1 = g[1].mean()
    mode_0 = g.agg({0: mode})[0]
    mode_1 = g.agg({1: mode})[1]
    S_0 = g.agg({0:entropy})[0]
    S_1 = g.agg({1:entropy})[1]

    #kde = g.apply(kernel_estimates, (hor_steps, vert_steps))

    ratio_10 = (high_10_1 - low_10_1) / (high_10_0 - low_10_0)
    ratio_20 = (high_20_1 - low_20_1) / (high_20_0 - low_20_0)
    mean_std_0 = mean_0 / std_0
    mean_std_1 = mean_1 / std_1

    iqr_0 = g[0].quantile(0.75) - g[0].quantile(0.25)
    iqr_1 = g[1].quantile(0.75) - g[1].quantile(0.25)


    corr = g[[0, 1]].corr().loc[(slice(None), 0), 1].droplevel(1)

    features = pd.DataFrame(data={"high_10_0": high_10_0, "high_10_1": high_10_1,
                                       "high_20_0": high_20_0, "high_20_1": high_20_1,
                                       "high_50_0": high_50_0, "high_50_1": high_50_1,
                                       "low_10_0": low_10_0, "low_10_1": low_10_1,
                                       "low_20_0": low_20_0, "low_20_1": low_20_1,
                                       "low_50_0": low_50_0, "low_50_1": low_50_1,
                                       "mean_0": mean_0, "mean_1": mean_1,
                                       "ratio_10": ratio_10, "ratio_20": ratio_20,
                                       "mean_std_0": mean_std_0, "mean_std_1": mean_std_1,
                                       "mode_0": mode_0, "mode_1": mode_1,
                                       "S_0": S_0, "S_1": S_1,
                                       "iqr_0":iqr_0,"iqr_1":iqr_1,
                                       "corr": corr
                                       })

    features = pd.merge(features,mpv,left_index=True,right_index=True,how="inner")
    features = pd.merge(features,pct,left_index=True,right_index=True,how="inner")
    features = pd.merge(features,pdw,left_index=True,right_index=True,how="inner")
    features = pd.merge(features,plt_count,left_index=True,right_index=True,how="inner")

    # Remove all infinity values.
    for c in features.columns:
        features.loc[features[c] == np.inf, c] = features.loc[features[c] < np.inf, c].max() * 1.5
        features.loc[features[c] == -np.inf, c] = features.loc[features[c] > -np.inf, c].min() * 1.5

    print("Not computing KDE !")
    #features = pd.merge(features, kde, left_index=True, right_index=True,how="inner")

    features = features.dropna(how="any",axis=0)

    return features

def build_features(df, sys_phen, train_IDs, hor_steps ,vert_steps, plot=False, save_pca=False,out_folder="/home/hv270/data"):
    #print("Building features %d" % os.getpid())
    features = dict()

    if plot:
        k = 1
        fig = plt.figure(figsize=(10, 8))

    exps = df.exp.unique().tolist()

    PLT_cond = (df.PLT > 0)

    d = df.loc[PLT_cond]

    def sample(x):
        return x[selected_columns].sample(3000, replace=True)

    X_sample = d.loc[d.ID.isin(train_IDs)].groupby(["ID", "exp"]).apply(sample).reset_index()

    # Features from each experiment
    X_s = []
    print("Features from each experiment")
    for exp in exps:

        X = X_sample.loc[X_sample['exp'] == exp, selected_columns].copy()

        #normalizer = QuantileTransformer(output_distribution="normal").fit(X)
        #normalizer = PowerTransformer().fit(X)
        normalizer = RobustScaler().fit(X)
        X = normalizer.transform(X)

        print("Fitting PCA %s" % exp)
        pca = PCA(n_components=2).fit(X)
        for i in range(2):
            pca.components_[i] /= np.sign(pca.components_[i, 0])

        if save_pca:
            print("Saving PCA of %s" % exp)
            pkl_filename = "%s/sysmex_pca_%s.pkl" % (out_folder,exp)
            with open(pkl_filename, 'wb') as file:
                pickle.dump(pca, file)

        d.loc[d.exp == exp, selected_columns] = normalizer.transform(d.loc[d.exp == exp, selected_columns])

        X_t = pca.transform(d.loc[d.exp == exp, selected_columns].values)
        X = pd.DataFrame(data=X_t, index=d.loc[d.exp == exp].index)
        X["ID"] = d.loc[d.exp == exp, "ID"]

        mpv = sys_phen.loc[sys_phen.exp == exp].groupby("ID")["MPV"].mean()
        plt_count = sys_phen.loc[sys_phen.exp == exp].rename(columns={"PLT_count":"PLT"}).groupby("ID")["PLT"].mean()
        pdw = sys_phen.loc[sys_phen.exp == exp].groupby("ID")["PDW"].mean()
        pct = sys_phen.loc[sys_phen.exp == exp].groupby("ID")["PCT"].mean()

        features[exp] = get_features_per_ID(X,mpv,plt_count,pdw,pct,hor_steps,vert_steps)

        X["exp"] = exp
        X_s.append(X)

        if plot:
            ax = fig.add_subplot(2, 2, k)
            k += 1
            ax.bar(x=np.arange(len(selected_columns)) - 0.4, height=pca.components_[0], width=0.2,
                   label="First component (%.2f)" % pca.explained_variance_ratio_[0])
            ax.bar(x=np.arange(len(selected_columns)) - 0.2, height=pca.components_[1], width=0.2,
                   label="Second component (%.2f)" % pca.explained_variance_ratio_[1])
            ax.set_xticks(np.arange(len(selected_columns)) - 0.3)
            ax.set_xticklabels([initials(name) for name in selected_columns])
            ax.set_title("PCA : %s samples" % exp.upper())
            ax.legend()

    features = pd.concat([features[e].add_suffix("_%s" % e) for e in exps], axis=1, sort=True)

    if plot:
        plt.tight_layout()

    assert features.shape[1] == len(set(features.columns.tolist()))

    features -= features.loc[train_IDs].mean(axis=0)
    features /= features.loc[train_IDs].std(axis=0)

    return features, pd.concat(X_s, axis=0, sort=True)
