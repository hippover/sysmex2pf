from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class predictor():
    def __init__(self,weights,pheno_name="pheno_name"):
        self.w = weights
        self.pheno_name = pheno_name

    def select_best_params(self,features,Y_,sys_phen):

        n_iter = 10

        scores = np.zeros((len(self.n_comps_pls),len(self.n_comps_pca),len(self.adjust_y)))
        scores_lm = np.zeros((len(self.n_comps_pls),len(self.n_comps_pca),len(self.adjust_y)))

        for i in range(len(self.n_comps_pls)):
            for j in range(len(self.n_comps_pca)):
                for k in range(len(self.adjust_y)):
                    score = np.zeros(n_iter)
                    score_lm = np.zeros(n_iter)
                    for iter_ in range(n_iter):

                        X_true = features.copy()
                        samples_OK = self.w.loc[self.w > 0].index.tolist()
                        X_true = X_true.loc[X_true.index.isin(samples_OK)]
                        Y_true = Y_.loc[X_true.index].copy()

                        preds, preds_lm = self.predict_in_splits(X_true,Y_true,sys_phen,
                                                                 self.adjust_y[k],
                                                                 self.n_comps_pls[i],
                                                                 self.n_comps_pca[j])

                        score[iter_] = r2_score(Y_true,preds)
                        score_lm[iter_] = r2_score(Y_true,preds_lm)

                    scores[i,j,k] = np.mean(score)
                    scores_lm[i,j,k] = np.mean(score_lm)

        self.best_only_loadings = np.max(scores) < np.max(scores_lm)
        if self.best_only_loadings:
            scores = scores_lm
        best_params = np.unravel_index(np.argmax(scores),np.shape(scores))
        self.best_n_comps_pls = self.n_comps_pls[best_params[0]]
        self.best_n_comps_pca = self.n_comps_pca[best_params[1]]
        self.best_adjust_y = self.adjust_y[best_params[2]]
        self.best_score = np.max(scores)
        print("==== %s ====" % self.pheno_name)
        print("Obtained maximal R^2 of %.3f" % np.max(scores))
        print("loadings only = %s " % self.best_only_loadings)
        print("n_comps_pca = %d " % self.best_n_comps_pca)
        print("n_comps_pls = %d " % self.best_n_comps_pls)
        print("adjust y = %s" % self.best_adjust_y)

    def predict_in_splits(self,X_true,Y_true,sys_phen,adjust_y,n_comps_pls,n_comps_pca):

        preds = pd.Series(data=np.zeros(X_true.shape[0]),index=X_true.index)
        preds_lm = pd.Series(data=np.zeros(X_true.shape[0]),index=X_true.index)
        skf = StratifiedKFold(n_splits=12)

        for train, test in skf.split(X_true,Y_true > np.median(Y_true)):

            X_ = X_true.copy()
            Y_ = Y_true.copy()

            index_train = X_true.index[train]
            index_test = X_true.index[test]

            X_train = X_true.loc[index_train]
            X_test = X_true.loc[index_test]

            self.train(X_train,Y_.loc[index_train],sys_phen.loc[index_train],adjust_y,n_comps_pls,n_comps_pca)
            preds.loc[index_test], preds_lm.loc[index_test] = self.predict(X_test,sys_phen.loc[index_test],adjust_y,n_comps_pls,n_comps_pca)

        return preds, preds_lm

    def train(self,X_t,Y_t,sys_phen,adjust_y,n_comps_pls,n_comps_pca):

        X_ = X_t.copy()
        Y_ = Y_t.copy()

        self.lm_y = LinearRegression().fit(sys_phen[["PLT_count","MPV","PDW","PCT"]],Y_)
        if adjust_y:
            sys_pred = self.lm_y.predict(sys_phen[["PLT_count","MPV","PDW","PCT"]])
        else:
            sys_pred = np.zeros(X_.shape[0])
        Y_ = Y_-sys_pred

        self.pca = PCA(n_components=n_comps_pca)
        self.pca.fit(X_)

        X_load = pd.DataFrame(data=self.pca.transform(X_),index=X_.index)
        self.lr = LinearRegression().fit(sys_phen[["PLT_count","MPV","PDW","PCT"]],X_load)
        X_load = X_load - self.lr.predict(sys_phen.loc[X_.index,["PLT_count","MPV","PDW","PCT"]])

        X_ = pd.DataFrame(data=self.pca.inverse_transform(X_load),index=X_.index)
        self.good_indices = np.where(np.std(X_,axis=0) > 0.001)[0]

        self.pls = PLSRegression(n_components=n_comps_pls).fit(X_.iloc[:,self.good_indices],Y_)
        self.lm = LinearRegression().fit(X_load,Y_)

    def train_on_best_params(self,X_t,Y_t,sys_phen):
        self.train(X_t,Y_t,
                sys_phen,self.best_adjust_y,
                self.best_n_comps_pls,
                self.best_n_comps_pca)

    def predict(self,X_,sys_phen,adjust_y,n_comps_pls,n_comps_pca):
        if adjust_y:
            sys_pred = self.lm_y.predict(sys_phen[["PLT_count","MPV","PDW","PCT"]])
        else:
            sys_pred = np.zeros(X_.shape[0])

        X_load = pd.DataFrame(data=self.pca.transform(X_),index=X_.index)
        X_load = X_load - self.lr.predict(sys_phen.loc[X_.index,["PLT_count","MPV","PDW","PCT"]])

        X_ = pd.DataFrame(data=self.pca.inverse_transform(X_load),index=X_.index)

        preds = pd.Series(data=self.pls.predict(X_.iloc[:,self.good_indices])[:,0] + sys_pred,index=X_.index)
        preds_lm = pd.Series(data=self.lm.predict(X_load) + sys_pred,index=X_.index)

        return preds, preds_lm

    def predict_with_best(self,X_,sys_phen):
        pred_pls, pred_loading = self.predict(X_,sys_phen,
                                              self.best_adjust_y,
                                              self.best_n_comps_pls,
                                              self.best_n_comps_pca)
        if self.best_only_loadings:
            return pred_loading
        else:
            return pred_pls

    def predict_in_splits_with_best(self,X_true,Y_true,sys_phen):

        pred = self.predict_in_splits(X_true,
                                      Y_true,
                                      sys_phen,
                                      self.best_adjust_y,
                                      self.best_n_comps_pls,
                                      self.best_n_comps_pca)

        if self.best_only_loadings:
            return pred[1]
        else:
            return pred[0]

class KDE_predictor(predictor):
    def __init__(self,weights,pheno_name="pheno_name"):
        super().__init__(weights,pheno_name+"_KDE")
        self.n_comps_pls = [1,2]
        self.n_comps_pca = [3,4,5]
        self.adjust_y = [True,False]

class agg_features_predictor(predictor):
    def __init__(self,weights,pheno_name="pheno_name"):
        super().__init__(weights,pheno_name+"_agg")
        self.n_comps_pls = [1,2]
        self.n_comps_pca = [3,4,5]
        self.adjust_y = [True,False]
