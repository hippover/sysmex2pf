import importlib
from .predictors import KDE_predictor, agg_features_predictor
from .build_sysmex_features import *
from sklearn.feature_selection import f_regression
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import AxesGrid
from sklearn.model_selection import StratifiedKFold
import time
import itertools
from multiprocessing import Pool
from sklearn.metrics import r2_score
import scipy
import os
import pickle
import tqdm
from functools import partial

n_processes = 30

def get_good_ids():
    with open("%s/good_ids.txt" % out_folder) as f:
        good_ids = f.readlines()
        good_ids = [g[:-1] for g in good_ids]
        return good_ids

sysmex_features = ['MPV_wb', 'PCT_wb', 'PLT_wb', 'PDW_wb']

def train_on_one_split(sys_df, sys_phen, all_fc_pheno, sample_weights, train_ids,
                       good_ids, hor_steps=30, vert_steps=15, plot=True):
    """
    Performs trainning just on one splits (without adjustments, very basic pipeline)
    and plots various checks regarding correlations
    :param sys_df: Sysmex measurements
    :param sys_phen: Sysmex phenotypes
    :param all_fc_pheno: FC phenotypes
    :param sample_weights: 1 / 0
    :param train_ids: train set
    :param good_ids: all ids
    :param hor_steps: number of horizontal (PC1) steps to compute densities
    :param vert_steps: number of vertical (PC2) steps
    :param plot: whether or not to do the plots
    :return: nothing
    """
    print("Building features")
    norm_features, X = build_features(sys_df.loc[sys_df.ID.isin(good_ids)].copy(),
                                      sys_phen.loc[sys_phen.ID.isin(good_ids)].copy(),
                                      train_ids, hor_steps, vert_steps,
                                      plot=plot, save_pca=False)

    print("Look for regressions")
    # P-value of regression
    reg_p = pd.DataFrame(index=norm_features.columns)
    reg_b = pd.DataFrame(index=norm_features.columns)

    for c in sample_weights.columns:
        if "wass" in c and "norm" in c:
            reg_p[c] = -np.log10(f_regression(norm_features.loc[good_ids].loc[(sample_weights[c] > 0)],
                                              all_fc_pheno.loc[good_ids].loc[(sample_weights[c] > 0), c],
                                              center=True)[1])
            for f in norm_features.columns:
                reg_b.loc[f, c] = pearsonr(norm_features.loc[good_ids].loc[(sample_weights[c] > 0), f],
                                           all_fc_pheno.loc[good_ids].loc[(sample_weights[c] > 0), c]
                                           )[0]
    reg_p["max"] = reg_p.max(axis=1)
    reg_p["argmax"] = reg_p.idxmax(axis=1)
    reg_p = reg_p.sort_values(by="max", ascending=False)
    print(reg_p[["argmax", "max"]].head(10))

    if plot:
        print("Plot correlations")
        kde_b = reg_b.loc[reg_b.index.str.contains("KDE")]
        kde_b_adp = kde_b[[c for c in kde_b.columns if "adp" in c]].mean(axis=1)
        kde_b_crp = kde_b[[c for c in kde_b.columns if "crp" in c]].mean(axis=1)

        fig = plt.figure(figsize=(12, 3.5))
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(2, 4),
                        axes_pad=0.5,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.1)

        adp = np.reshape(kde_b_adp.values, (4, hor_steps, vert_steps))
        crp = np.reshape(kde_b_crp.values, (4, hor_steps, vert_steps))

        for i in range(4):
            cm = grid[i].imshow(np.rot90(adp[(i - 1) % 4]), cmap=plt.cm.seismic, vmin=-.7, vmax=.7,
                                extent=[-7, 7, -3.5, 3.5], aspect="equal")
            cm = grid[i + 4].imshow(np.rot90(crp[(i - 1) % 4]), cmap=plt.cm.seismic, vmin=-.7, vmax=.7,
                                    extent=[-7, 7, -3.5, 3.5], aspect="equal")

        grid[0].set_ylabel("PC2")
        grid[4].set_ylabel("PC2")
        exps = ["WB", "Rest", "ADP", "CRP"]
        for k in [4, 5, 6, 7]:
            grid[k].set_xlabel("PC1")
            grid[k - 4].set_title(exps[k - 4])

        cbar = grid.cbar_axes[0].colorbar(cm)
        cbar.ax.set_ylabel("mean Pearson correlation")

        cbar.ax.set_yticks([-.7, 0, .7])
        plt.suptitle("Mean Pearson correlation with ADP (first row) and CRP (second row) phenotypes", fontsize=14)
        cbar.ax.set_yticklabels(['PLT density higher \nfor low reactivity', 'equal densities',
                                 'PLT density higher \nfor high reactivity'])

def get_bad_cols(features,train):
  bad_cols = [c for c in features.columns if features.loc[train, c].nunique() < 5]
  bad_cols += [c for c in features.columns if "KDE" in c and features.loc[train, c].std() < 2.5e-3]
  return bad_cols


def build_targets_(l):
    """
    :param l: list containing everything needed. Made like that to run in parallel
    :return:
    """
    time, train, test, all_fc_pheno, sys_df, sys_phen, good_ids = l
    good_ids = train + test
    targets = all_fc_pheno[[c for c in all_fc_pheno.columns if c.upper() != c]].copy().loc[good_ids]

    return targets


def make_predictions(l):
    """
    :param l: list containing everything needed. Made like that to run in parallel
    :return:
    """
    time, train, test, all_fc_pheno, sys_df, sys_phen, norm_features, norm_targets, w, assays, adjust,out_folder = l

    assert norm_features.shape[1] == len(set(norm_features.columns))

    valid_cols = []
    print(assays)
    for assay in assays:
        valid_cols += [c for c in norm_features.columns.tolist() if assay in c]

    assert len(valid_cols) == len(set(valid_cols))

    return train_with_split(norm_features[valid_cols], norm_targets, train, w, out_folder,
                            adjust=adjust, save_preds=False)


def train_on_splits(sys_df, sys_phen, all_fc_pheno, sample_weights, good_ids,hor_steps,vert_steps, out_folder,
                    assays=("wb", "rest", "adp", "crp"), n_iter=1, n_splits=10, adjust=True):
    """
    :param sys_df:
    :param sys_phen:
    :param all_fc_pheno:
    :param sample_weights:
    :param good_ids:
    :param assays: tuple of assays to use
    :param n_iter:
    :param n_splits:
    :return:
    """
    pheno_columns = [c for c in all_fc_pheno.columns if c.upper() != c]

    train_ = []
    test_ = []
    ts = time.strftime("%Y%m%d_%H_%M_%S", time.gmtime())
    times_ = []
    sys_ = []
    sys_phen_ = []
    fc_p_ = []
    w_ = []
    assays_ = []
    adjust_ = []
    good_ = []
    hor_ = []
    vert_ = []
    out_ = []

    for i in range(n_iter):
        j = 0
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i * n_splits + j)
        for train_index, test_index in skf.split(np.zeros(len(good_ids)), np.zeros(len(good_ids))):
            train_IDs = [good_ids[k] for k in train_index]
            test_IDs = [good_ids[k] for k in test_index]

            train_.append(train_IDs)
            test_.append(test_IDs)
            times_.append("%s_%d" % (ts, i * n_splits + j))
            sys_.append(sys_df.loc[sys_df.ID.isin(good_ids)].copy())
            sys_phen_.append(sys_phen.loc[sys_phen.ID.isin(good_ids)].copy())
            fc_p_.append(all_fc_pheno.loc[good_ids, pheno_columns].copy())  # Remove all copy()
            w_.append(sample_weights.loc[good_ids].copy())
            assays_.append(assays)
            adjust_.append(adjust)
            good_.append(good_ids)
            hor_.append(hor_steps)
            vert_.append(vert_steps)
            out_.append(out_folder)
            j += 1

    pool = Pool(n_processes)
    features_list = pool.map(build_features_, itertools.zip_longest(times_, train_, test_, fc_p_, sys_, sys_phen_, good_,hor_,vert_))
    pool.close()
    pool.join()

    pool = Pool(n_processes)
    targets_list = pool.map(build_targets_, itertools.zip_longest(times_, train_, test_, fc_p_, sys_, sys_phen_, good_))
    pool.close()
    pool.join()

    # TODO
    # Only here do we filter features using assays, be we could do it earlier and not generate them
    pool = Pool(n_processes)
    res = pool.map(make_predictions,
                   itertools.zip_longest(times_, train_, test_, fc_p_, sys_, sys_phen_, features_list,
                                         targets_list, w_, assays_, adjust_,out_))
    pool.close()
    pool.join()

    recap_scores, predictions = get_recap_scores(res, sample_weights, n_splits, n_iter, test_,good_ids,adjust=adjust)
    recap_scores_features, predictions_features = get_recap_scores_features_adjustment(res, sample_weights, n_splits, n_iter, test_,good_ids,adjust=adjust)

    return recap_scores, predictions, recap_scores_features, predictions_features

def get_recap_scores(res, sample_weight, n_splits, n_iter, test_, good_ids,adjust=True):
    """
    Computes R^2 statistics using test samples from every iteration / split
    :param res:
    :param sample_weight:
    :param n_splits:
    :param n_iter:
    :param test_:
    :param adjust:
    :return: a summary dataframe, and test predictions
    """
    predictions = None
    scores_sum = np.zeros((res[0]["residuals"].shape[1], n_iter))  # sysmex-based prediction + residuals vs target
    p_val = np.zeros((res[0]["residuals"].shape[1], n_iter))  # p-value of target/prediction correlation
    pearson = np.zeros((res[0]["residuals"].shape[1], n_iter))  # pearson of target/prediction

    for k in range(n_iter):
        # We create one dataframe per iteration
        target_ = pd.DataFrame(index=good_ids, columns=res[0]["targets"].columns.tolist())
        pred_ = pd.DataFrame(index=good_ids, columns=res[0]["predictions"].columns.tolist())
        for j in range(n_splits):
            l = k * n_splits + j
            # And fill it with the test sets of each split
            target_.loc[test_[l], res[l]["targets"].columns] = res[l]["targets"].loc[test_[l]]
            pred_.loc[test_[l], res[l]["predictions"].columns] = res[l]["predictions"].loc[test_[l]]
        predictions = pred_

        for i in range(pred_.shape[1]):

            scores_sum[i, k] = r2_score(target_.iloc[:, i], pred_.iloc[:, i],
                                        sample_weight.loc[good_ids].iloc[:, i])
            pearson[i,k], p_val[i,k] = scipy.stats.pearsonr(target_.loc[sample_weight.loc[good_ids,pred_.columns[i]] > 0].iloc[:, i],
                                                pred_.loc[sample_weight.loc[good_ids,pred_.columns[i]] > 0].iloc[:, i])

            if scores_sum[i, k] > 0.2:

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot([target_.loc[sample_weight.loc[good_ids].iloc[:, i] > 0].iloc[:, i].min(),
                         target_.loc[sample_weight.loc[good_ids].iloc[:, i] > 0].iloc[:, i].max()],
                        [target_.loc[sample_weight.loc[good_ids].iloc[:, i] > 0].iloc[:, i].min(),
                         target_.loc[sample_weight.loc[good_ids].iloc[:, i] > 0].iloc[:, i].max()],
                        linestyle=":", c="red")
                ax.scatter(target_.loc[sample_weight.loc[good_ids].iloc[:, i] > 0].iloc[:, i],
                           pred_.loc[sample_weight.loc[good_ids].iloc[:, i] > 0].iloc[:, i])
                ax.set_xlabel("True values")
                ax.set_ylabel("Predicted values")
                ax.set_title("%s : score = %.2f" % (pred_.columns.tolist()[i],scores_sum[i, k]))

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_title("Target vs pred (%d points)" % target_.loc[sample_weight.loc[good_ids].iloc[:, i] > 0].iloc[:, i].shape[0])
                ax.hist(target_.loc[sample_weight.loc[good_ids].iloc[:, i] > 0].iloc[:, i],label="target",alpha=0.7)
                ax.hist(pred_.loc[sample_weight.loc[good_ids].iloc[:, i] > 0].iloc[:, i],label="pred",alpha=0.7)
                ax.legend()

                plt.savefig("/home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/hv270/predictions/nice_pred_%s.png" % pred_.columns.tolist()[i])

    mean_scores_sum = np.mean(scores_sum, axis=1)

    mean_p_val = np.mean(p_val,axis=1)
    mean_pearson = np.mean(pearson,axis=1)

    recap_scores = pd.DataFrame(index=target_.columns)
    recap_scores["Overall"] = mean_scores_sum
    recap_scores["p-value"] = mean_p_val
    recap_scores["pearson"] = mean_pearson

    return recap_scores, predictions  # Predicted phenotypes

def get_recap_scores_features_adjustment(res, sample_weight, n_splits, n_iter, test_, good_ids,adjust=True):
    """
    Computes R^2 statistics of features adjustments using test samples from every iteration / split
    :param res:
    :param sample_weight:
    :param n_splits:
    :param n_iter:
    :param test_:
    :param adjust:
    :return: a summary dataframe, and test predictions
    """
    common_features = res[0]["adjusted_features"].columns
    print("Start with %d features" % len(common_features))
    for i in range(len(res)):
        common_features = list(set(common_features).intersection(res[i]["adjusted_features"].columns))
    print("End with %d features, shared across all splits" % len(common_features))

    predictions = None
    scores_sum = np.zeros((len(common_features), n_iter))  # sysmex-based prediction + residuals vs target
    p_val = np.zeros((len(common_features), n_iter))  # p-value of target/prediction correlation

    for k in range(n_iter):
        # We create one dataframe per iteration
        target_ = pd.DataFrame(index=good_ids, columns=common_features)
        pred_ = pd.DataFrame(index=good_ids, columns=common_features)
        for j in range(n_splits):
            l = k * n_splits + j
            # And fill it with the test sets of each split
            target_.loc[test_[l], common_features] = res[l]["features"].loc[test_[l], common_features]
            pred_.loc[test_[l], common_features] = res[l]["features"].loc[test_[l], common_features] - res[l]["adjusted_features"].loc[test_[l], common_features]
        predictions = pred_

        for i in range(len(common_features)):

            scores_sum[i, k] = r2_score(target_.iloc[:, i], pred_.iloc[:, i])
            p_val[i,k] = scipy.stats.pearsonr(target_.iloc[:, i],
                                                pred_.iloc[:, i])[1]

            if scores_sum[i, k] > 0.2:

                fig = plt.figure(figsize=(5,8),dpi=200)
                ax = fig.add_subplot(121)
                ax.plot([target_.iloc[:, i].min(),
                         target_.iloc[:, i].max()],
                        [target_.iloc[:, i].min(),
                         target_.iloc[:, i].max()],
                        linestyle=":", c="red")
                ax.scatter(target_.iloc[:, i],
                             pred_.iloc[:, i])
                ax.set_xlabel("True values")
                ax.set_ylabel("Predicted values")
                ax.set_title("%s : score = %.2f" % (common_features[i],scores_sum[i, k]))

                ax = fig.add_subplot(122)
                ax.set_title("Target vs pred (%d points)" % target_.iloc[:, i].shape[0])
                ax.hist(target_.iloc[:, i],label="target",alpha=0.7)
                ax.hist(pred_.iloc[:, i],label="pred",alpha=0.7)
                ax.legend()

                plt.savefig("/home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/hv270/predictions/nice_pred_%s.png" % common_features[i])

            elif scores_sum[i, k] < -2:

                fig = plt.figure(figsize=(5,8),dpi=200)
                ax = fig.add_subplot(211)
                ax.plot([target_.iloc[:, i].min(),
                         target_.iloc[:, i].max()],
                        [target_.iloc[:, i].min(),
                         target_.iloc[:, i].max()],
                        linestyle=":", c="red")
                ax.scatter(target_.iloc[:, i],
                             pred_.iloc[:, i])
                ax.set_xlabel("True values")
                ax.set_ylabel("Predicted values")
                ax.set_title("%s : score = %.2f" % (common_features[i],scores_sum[i, k]))

                ax = fig.add_subplot(212)
                ax.set_title("Target vs pred (%d points)" % target_.iloc[:, i].shape[0])
                ax.hist(target_.iloc[:, i],label="target",alpha=0.7)
                ax.hist(pred_.iloc[:, i],label="pred",alpha=0.7)
                ax.legend()

                plt.savefig("/home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/hv270/predictions/bad_pred_%s.png" % common_features[i])

    mean_scores_sum = np.mean(scores_sum, axis=1)

    mean_p_val = np.mean(p_val,axis=1)

    recap_scores = pd.DataFrame(index=common_features)
    recap_scores["Overall"] = mean_scores_sum
    recap_scores["p-value"] = mean_p_val

    return recap_scores, predictions  # Predicted phenotypes

def train_predictor_proxy(pheno,sample_weights,features_KDE,features_agg,sys_phen,y):
    pred_KDE = KDE_predictor(sample_weights[pheno],pheno)
    pred_KDE.select_best_params(features_KDE,y[pheno],sys_phen)
    pred_KDE.train_on_best_params(features_KDE,y[pheno],sys_phen)

    pred_agg = agg_features_predictor(sample_weights[pheno],pheno)
    pred_agg.select_best_params(features_agg,y[pheno],sys_phen)
    pred_agg.train_on_best_params(features_agg,y[pheno],sys_phen)

    prediction_agg = pred_agg.predict_in_splits_with_best(features_agg,y[pheno],sys_phen)
    prediction_KDE = pred_KDE.predict_in_splits_with_best(features_KDE,y[pheno],sys_phen)

    #return a weighted average of the two predictions, overweighting the best one
    agg = pred_agg.best_score**2
    kde = pred_KDE.best_score**2
    prediction = prediction_agg*agg + prediction_KDE*kde
    prediction /= (agg + kde)

    return pheno, pred_KDE, pred_agg,prediction

def train_and_export(sys_df,sys_phen,all_fc_pheno,sample_weights,good_ids,hor_steps,vert_steps,out_folder,adjust=True):
    """
    Trains predictors on all donors and exports them, as well as needed information to use the predictors on other data
    :param sys_df:
    :param sys_phen:
    :param all_fc_pheno:
    :param sample_weights:
    :param good_ids:
    :param adjust:
    :return:
    """
    features, X = build_features(sys_df.loc[sys_df.ID.isin(good_ids) & sys_df.exp.isin(["wb"])].copy(),
                                    sys_phen.loc[sys_phen.ID.isin(good_ids)].copy(),
                                    good_ids,
                                    hor_steps, vert_steps,
                                    plot=True,
                                    save_pca=True,
                                    out_folder=out_folder
                                    )
    sys_phen = sys_phen.loc[sys_phen.exp == "wb"]
    sys_phen = sys_phen.set_index("ID")

    features = features.loc[good_ids]
    targets = all_fc_pheno.copy().loc[good_ids]

    features_KDE = features[sorted([c for c in features.columns if "wb" in c and "KDE" in c])]
    features_KDE = features_KDE.loc[good_ids].dropna(axis=0,how='any')
    print(features_KDE.head())

    features_no_KDE = features[[c for c in features.columns if "wb" in c and "KDE" not in c]]
    features_no_KDE = features_no_KDE[features_no_KDE.columns[:-4]] # drop Sysmex features
    features_no_KDE = features_no_KDE[sorted(features_no_KDE.columns)]
    features_no_KDE = features_no_KDE.loc[good_ids].dropna(axis=0,how='any')
    print(features_no_KDE.head())

    pool = Pool(n_processes)
    prediction_results = pd.DataFrame(index=targets.columns)
    prediction_results["Overall"] = 0.
    prediction_results["pearson"] = 0.

    if not os.path.exists("%s/new_predictors/" % out_folder):
        os.mkdir("%s/new_predictors/" % out_folder)

    print("Training predictors")
    # WARNING : only N_PROCESSES columns
    prediction_all = pd.DataFrame(index=targets.index)
    for p_preds in tqdm.tqdm(pool.imap_unordered(partial(train_predictor_proxy,
                                                sample_weights=sample_weights,
                                                features_KDE=features_KDE.loc[good_ids],
                                                features_agg=features_no_KDE.loc[good_ids],
                                                sys_phen = sys_phen.loc[good_ids],
                                                y = all_fc_pheno.loc[good_ids]),
                                                targets.columns),total=targets.shape[1]):
        # pred_KDE and pred_agg are predictors, not predictions
        pheno, pred_KDE, pred_agg, prediction = p_preds
        prediction_all[pheno] = prediction
        f = open("%s/new_predictors/%s_KDE.pkl" % (out_folder,pheno),"wb")
        pickle.dump(pred_KDE,f)
        f.close()
        f = open("%s/new_predictors/%s_agg.pkl" % (out_folder,pheno),"wb")
        pickle.dump(pred_agg,f)
        f.close()
        prediction_results.loc[pheno,"Overall"] = r2_score(all_fc_pheno.loc[prediction.index,pheno],prediction)
        prediction_results.loc[pheno,"pearson"] = pearsonr(all_fc_pheno.loc[prediction.index,pheno],prediction)[0]

    print(prediction_results.sort_values("Overall",ascending=False).head(30))
    prediction_results.to_csv("%s/prediction_results.csv" % out_folder)
    prediction_all.to_csv("%s/predicted_phenotypes.csv" % out_folder)

    features_KDE.to_csv("%s/features_KDE.csv" % out_folder)
    features_no_KDE.to_csv("%s/features_no_KDE.csv" % out_folder)
