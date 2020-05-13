from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from scipy.special import comb
from scipy.stats import pearsonr
from tqdm.notebook import tqdm

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

        scores = -np.inf*np.ones((len(self.n_comps_pls),len(self.n_comps_pca),len(self.adjust_y)))
        scores_lm = -np.inf*np.ones((len(self.n_comps_pls),len(self.n_comps_pca),len(self.adjust_y)))
        BIC_pls = np.inf*np.ones((len(self.n_comps_pls), len(self.n_comps_pca), len(self.adjust_y)))
        BIC_lm = np.inf*np.ones((len(self.n_comps_pls), len(self.n_comps_pca), len(self.adjust_y)))

        progress_bar = tqdm(total=n_iter*len(self.n_comps_pls)*len(self.n_comps_pca)*len(self.adjust_y))
        for i in range(len(self.n_comps_pls)):
            for j in range(len(self.n_comps_pca)):
                for k in range(len(self.adjust_y)):
                    score = np.zeros(n_iter)
                    score_lm = np.zeros(n_iter)

                    for iter_ in range(n_iter):

                        X_true = features.copy()
                        Y_true = Y_.loc[X_true.index].copy()

                        preds, preds_lm = self.predict_in_splits(X_true,Y_true,sys_phen,
                                                                 self.adjust_y[k],
                                                                 self.n_comps_pls[i],
                                                                 self.n_comps_pca[j])

                        score[iter_] = r2_score(Y_true,preds)
                        score_lm[iter_] = r2_score(Y_true,preds_lm)

                        progress_bar.update(1)

                    if self.n_comps_pls[i] >= len(self.good_columns):
                        scores[i, j, k] = np.mean(score)
                    scores_lm[i, j, k] = np.mean(score_lm)

                    # To compute the BIC, we train on the whole dataset
                    self.train(X_true,Y_true,sys_phen,
                               self.adjust_y[k],
                               self.n_comps_pls[i],
                               self.n_comps_pca[j])
                    preds_pls, preds_lm = self.predict(X_true,sys_phen,self.adjust_y[k])
                    error_2_pls = np.std(Y_true - preds_pls)**2
                    error_2_lm = np.std(Y_true - preds_lm)**2

                    n_params = 0
                    # Adjust y
                    if self.adjust_y[k] == True:
                        n_params += 4
                    # PCA, we finally don't count it as it hasn't to do with Y
                    # Last term because principal components are orthogonal
                    # n_params += X_true.shape[1]*self.n_comps_pca[j] - comb(self.n_comps_pca[j],2)
                    # adjust loadings, we don't count them either
                    # n_params += 4*self.n_comps_pca[j]
                    # PLS
                    n_params_pls = n_params + len(self.good_columns)*self.n_comps_pls[i] + self.n_comps_pls[i] - comb(self.n_comps_pls[i],2)
                    # Only LM
                    n_params_lm = n_params + self.n_comps_pca[j] * self.n_comps_pls[i] + self.n_comps_pls[i]
                    if self.n_comps_pls[i] <= len(self.good_columns):
                        BIC_pls[i,j,k] = preds.shape[0]*np.log(np.mean(error_2_pls)) + n_params_pls*np.log(preds.shape[0])
                    BIC_lm[i,j,k] = preds.shape[0]*np.log(np.mean(error_2_lm)) + n_params_lm*np.log(preds.shape[0])

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
        else:
            raise Exception("choice_critetrion should be BIC or R2")

        self.best_n_comps_pls = self.n_comps_pls[best_params[0]]
        self.best_n_comps_pca = self.n_comps_pca[best_params[1]]
        self.best_adjust_y = self.adjust_y[best_params[2]]
        self.best_score = scores[best_params[0],best_params[1],best_params[2]]
        self.best_params_found = True

        print("==== %s ====" % self.pheno_name)
        print("Chosen with %s" % self.choice_criterion)
        print("Obtained R^2 of %.3f" % self.best_score)
        print("loadings only = %s " % self.best_only_loadings)
        print("n_comps_pca = %d " % self.best_n_comps_pca)
        print("n_comps_pls = %d " % self.best_n_comps_pls)
        print("adjust y = %s" % self.best_adjust_y)

    def predict_in_splits(self,X_true,Y_true,sys_phen,adjust_y,n_comps_pls,n_comps_pca):

        preds = pd.Series(data=np.zeros(X_true.shape[0]),index=X_true.index)
        preds_lm = pd.Series(data=np.zeros(X_true.shape[0]),index=X_true.index)
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
                       n_comps_pls,
                       n_comps_pca)
            preds.loc[index_test], preds_lm.loc[index_test] = self.predict(X_test,sys_phen.loc[index_test],adjust_y)

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
        #self.pca = PCA(n_components="mle")
        self.pca.fit(X_)
        #print("MLE n_components is %d" % self.pca.n_components_)

        X_load = pd.DataFrame(data=self.pca.transform(X_),index=X_.index)
        self.lr = LinearRegression().fit(sys_phen[["PLT_count","MPV","PDW","PCT"]],X_load)
        X_load = X_load - self.lr.predict(sys_phen.loc[X_.index,["PLT_count","MPV","PDW","PCT"]])

        X_ = pd.DataFrame(data=self.pca.inverse_transform(X_load),index=X_.index)
        self.good_columns = np.where(np.std(X_,axis=0) > 0.001)[0]

        #print(n_comps_pca,X_.shape[1],len(self.good_columns))

        self.pls = PLSRegression(n_components=min(n_comps_pls,len(self.good_columns))).fit(X_.iloc[:,self.good_columns],Y_)

        self.lm = LinearRegression().fit(X_load,Y_)
        self.best_params_set = False

    def train_on_best_params(self,X_t,Y_t,sys_phen):
        assert self.best_params_found == True
        self.train(X_t,Y_t,
                sys_phen,self.best_adjust_y,
                self.best_n_comps_pls,
                self.best_n_comps_pca)
        self.best_params_set = True

    def predict(self,X_,sys_phen,adjust_y): # ,n_comps_pls,n_comps_pca
        if adjust_y:
            sys_pred = self.lm_y.predict(sys_phen[["PLT_count","MPV","PDW","PCT"]])
        else:
            sys_pred = np.zeros(X_.shape[0])

        X_load = pd.DataFrame(data=self.pca.transform(X_),index=X_.index)
        X_load = X_load - self.lr.predict(sys_phen.loc[X_.index,["PLT_count","MPV","PDW","PCT"]])

        X_ = pd.DataFrame(data=self.pca.inverse_transform(X_load),index=X_.index)

        preds = pd.Series(data=self.pls.predict(X_.iloc[:,self.good_columns])[:,0] + sys_pred,index=X_.index)
        preds_lm = pd.Series(data=self.lm.predict(X_load) + sys_pred,index=X_.index)

        return preds, preds_lm

    def predict_with_best(self,X_,sys_phen):
        assert self.best_params_found == True
        assert self.best_params_set == True
        pred_pls, pred_loading = self.predict(X_,sys_phen,
                                              self.best_adjust_y)
        if self.best_only_loadings:
            return pred_loading
        else:
            return pred_pls

    def predict_in_splits_with_best(self,X_true,Y_true,sys_phen):
        assert self.best_params_found == True

        pred = self.predict_in_splits(X_true,
                                      Y_true,
                                      sys_phen,
                                      self.best_adjust_y,
                                      self.best_n_comps_pls,
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
        self.adjust_y = [True,False]

class agg_features_predictor(predictor):
    def __init__(self,pheno_name="pheno_name",criterion="R2"):
        super().__init__(pheno_name+"_agg",criterion)
        self.n_comps_pls = [1,2,3]
        self.n_comps_pca = [1,2,3,4]
        self.adjust_y = [True,False]
