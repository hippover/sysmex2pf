from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from scipy.special import comb
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingRegressor

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class predictor():
    def __init__(self,pheno_name="pheno_name", choice_criterion = "R2"):
        self.pheno_name = pheno_name
        self.best_params_found = False
        self.best_params_set = False
        self.choice_criterion = choice_criterion
        self.n_comps_pls = []
        self.n_comps_pca = []
        self.adjust_y = []

    def select_best_params(self,features,Y_,sys_phen):

        n_iter = 3

        scores = -np.inf*np.ones((len(self.n_comps_pca),len(self.adjust_y)))
        scores_lm = -np.inf*np.ones((len(self.n_comps_pca),len(self.adjust_y)))
        BIC_pls = np.inf*np.ones((len(self.n_comps_pca), len(self.adjust_y)))
        BIC_lm = np.inf*np.ones((len(self.n_comps_pca), len(self.adjust_y)))

        progress_bar = tqdm(total=len(self.n_comps_pls)*len(self.n_comps_pca)*len(self.adjust_y))
        for j in range(len(self.n_comps_pca)):
            for k in range(len(self.adjust_y)):

                if self.choice_criterion == "R2":
                    score_lm = np.zeros(n_iter)

                    for iter_ in range(n_iter):

                        X_true = features.copy()
                        Y_true = Y_.loc[X_true.index].copy()

                        preds_lm, pred_sys = self.predict_in_splits(X_true,Y_true,sys_phen,
                                                                 self.adjust_y[k],
                                                                 self.n_comps_pca[j])

                        score_lm[iter_] = r2_score(Y_true-pred_sys,preds_lm)

                    scores_lm[j, k] = np.mean(score_lm)
                elif self.choice_criterion == "BIC":
                    # To compute the BIC, we train on the whole dataset

                    X_true = features.copy()
                    Y_true = Y_.loc[X_true.index].copy()

                    self.train(X_true,Y_true,sys_phen,
                               self.adjust_y[k],
                               self.n_comps_pca[j])
                    preds_lm, preds_sys = self.predict(X_true,sys_phen,self.adjust_y[k])
                    error_2_lm = np.std((Y_true - preds_sys) - (preds_lm - preds_sys))**2
                    scores_lm[j, k] = r2_score(Y_true - preds_sys, preds_lm - preds_sys)

                    n_params = 0
                    # Adjust y
                    # if self.adjust_y[k] == True:
                    #    n_params += 4
                    # PCA, we finally don't count it as it hasn't to do with Y
                    # Last term because principal components are orthogonal
                    # n_params += X_true.shape[1]*self.n_comps_pca[j] - comb(self.n_comps_pca[j],2)
                    # adjust loadings, we don't count them either
                    # n_params += 4*self.n_comps_pca[j]
                    # Only LM
                    n_params_lm = 1 + self.n_comps_pca[j]*(features.shape[1] + 1)
                    BIC_lm[j,k] = preds_lm.shape[0]*np.log(np.mean(error_2_lm)) + n_params_lm*np.log(preds_lm.shape[0])

                progress_bar.update(1)

        progress_bar.close()

        if self.choice_criterion == "R2":
            self.best_only_loadings = np.max(scores) < np.max(scores_lm)
            if self.best_only_loadings:
                scores = scores_lm
            best_params = np.unravel_index(np.argmax(scores),np.shape(scores))

        elif self.choice_criterion == "BIC":
            self.best_only_loadings = np.min(BIC_lm) < np.min(BIC_pls)
            if self.best_only_loadings:
                BIC = BIC_lm
                scores = scores_lm
            else:
                BIC = BIC_pls
            best_params = np.unravel_index(np.argmin(BIC), np.shape(BIC))
            self.best_BIC = np.min(BIC)
        else:
            raise Exception("choice_critetrion should be BIC or R2")

        self.best_n_comps_pca = self.n_comps_pca[best_params[0]]
        self.best_adjust_y = self.adjust_y[best_params[1]]
        self.best_score = scores[best_params[0],best_params[1]]
        self.best_params_found = True

        print("==== %s ====" % self.pheno_name)
        print("Chosen with %s" % self.choice_criterion)
        print("Obtained R^2 of %.2f on residuals" % self.best_score)
        print("n_comps_pca = %d " % self.best_n_comps_pca)

    def predict_in_splits(self,X_true,Y_true,sys_phen,adjust_y,n_comps_pca):

        preds_lm = pd.Series(data=np.zeros(X_true.shape[0]),index=X_true.index)
        preds_sysmex = pd.Series(data=np.zeros(X_true.shape[0]), index=X_true.index)
        skf = StratifiedKFold(n_splits=12)

        for train, test in skf.split(X_true,Y_true > np.median(Y_true)):

            Y_ = Y_true.copy()

            index_train = X_true.index[train]
            index_test = X_true.index[test]

            X_train = X_true.loc[index_train].copy()
            X_test = X_true.loc[index_test].copy()

            self.train(X_train,
                       Y_.loc[index_train],
                       sys_phen.loc[index_train],
                       adjust_y,
                       n_comps_pca)
            preds_lm.loc[index_test], preds_sysmex.loc[index_test] = self.predict(X_test,sys_phen.loc[index_test],adjust_y)

        return preds_lm, preds_sysmex

    def train(self,X_t,Y_t,sys_phen,adjust_y,n_comps_pca):

        X_ = X_t.copy()
        Y_ = Y_t.copy()

        # Adjust phenotypes
        self.lm_y = LinearRegression().fit(sys_phen[["PLT_wb","MPV_wb","PDW_wb","PCT_wb"]],Y_)
        if adjust_y:
            sys_pred = self.lm_y.predict(sys_phen[["PLT_wb","MPV_wb","PDW_wb","PCT_wb"]])
        else:
            sys_pred = np.zeros(X_.shape[0])
        Y_ = Y_-sys_pred

        self.pca = PCA(n_components=n_comps_pca)

        # Option 1 : ajuster les loadings
        """
        self.pca.fit(X_)
        X_load = pd.DataFrame(data=self.pca.transform(X_),index=X_.index)
        self.lr = LinearRegression().fit(sys_phen[["PLT_count","MPV","PDW","PCT"]],X_load)
        X_load = X_load - self.lr.predict(sys_phen.loc[X_.index,["PLT_count","MPV","PDW","PCT"]])
        """

        #Option 2: ajuster les features
        self.lr = LinearRegression().fit(sys_phen[["PLT_wb","MPV_wb","PDW_wb","PCT_wb"]],X_)
        X_ = X_ - self.lr.predict(sys_phen.loc[X_.index,["PLT_wb","MPV_wb","PDW_wb","PCT_wb"]])
        #X_load = pd.DataFrame(data=self.pca.fit_transform(X_),index=X_.index)
        #
        #self.lm = LinearRegression().fit(X_load,Y_)
        self.lm = PLSRegression(n_components=n_comps_pca).fit(X_,Y_)

        self.best_params_set = False

    def train_on_best_params(self,X_t,Y_t,sys_phen):
        assert self.best_params_found == True
        self.train(X_t,Y_t,
                sys_phen,self.best_adjust_y,
                self.best_n_comps_pca)
        self.best_params_set = True

    def predict(self,X_,sys_phen,adjust_y): # ,n_comps_pls,n_comps_pca
        if adjust_y:
            sys_pred = self.lm_y.predict(sys_phen[["PLT_wb","MPV_wb","PDW_wb","PCT_wb"]])
        else:
            sys_pred = np.zeros(X_.shape[0])

        # Option 1
        """
        X_load = pd.DataFrame(data=self.pca.transform(X_),index=X_.index)
        X_load = X_load - self.lr.predict(sys_phen.loc[X_.index,["PLT_count","MPV","PDW","PCT"]])
        X_ = pd.DataFrame(data=self.pca.inverse_transform(X_load),index=X_.index)
        """
        # Option 2
        X_ = X_ - self.lr.predict(sys_phen.loc[X_.index,["PLT_wb","MPV_wb","PDW_wb","PCT_wb"]])
        #X_load = pd.DataFrame(data=self.pca.transform(X_),index=X_.index)
        #preds_lm = pd.Series(data=self.lm.predict(X_load) + sys_pred,index=X_.index)
        preds_lm = pd.Series(data=self.lm.predict(X_)[:,0] + sys_pred, index=X_.index)

        return preds_lm, sys_pred

    def predict_with_best(self,X_,sys_phen, return_sysmex = False):
        assert self.best_params_found == True
        assert self.best_params_set == True
        pred_loading, pred_sysmex = self.predict(X_,sys_phen,
                                              self.best_adjust_y)
        if return_sysmex:
            return pred_sysmex
        else:
            return pred_loading

    def predict_in_splits_with_best(self,X_true,Y_true,sys_phen):
        assert self.best_params_found == True

        pred = self.predict_in_splits(X_true,
                                      Y_true,
                                      sys_phen,
                                      self.best_adjust_y,
                                      self.best_n_comps_pca)
        self.best_params_set = True
        # Because predict_in_splits implies training,
        # this will be set to False but should be set back to True
        if self.best_only_loadings:
            return pred[1]
        else:
            return pred[0]

class KDE_predictor(predictor):
    def __init__(self,pheno_name="pheno_name",criterion="R2"):
        super().__init__(pheno_name+"_KDE",criterion)
        self.n_comps_pls = [1,2,3,4]
        self.n_comps_pca = [3,4,5,6,7]
        self.adjust_y = [True]

class agg_features_predictor(predictor):
    def __init__(self,pheno_name="pheno_name",criterion="R2"):
        super().__init__(pheno_name+"_agg",criterion)
        self.n_comps_pca = [1,2,3,4,5]
        self.adjust_y = [True]
