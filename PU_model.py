import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold

class PU_model:
    def __init__(self, model=SGDClassifier(loss='log', max_iter=10000, alpha=0.001),
                 estimator_type=1, weighted=False):
        """ model_s - модель g(x), оценивающая вероятность p(s=1|x),
        estimator_type принимает значения 1,2,3, сооствествующие способам расчета константы (self.const),
        weighted: False - алгоритм из второй главы(возвращает констату с), 
        True - из третьей, в котором множеству без меток (U) присваивается значение y=1,
        создается копия множества U, которому присваивается y=0, а затем алгоритм обучается заново 
        на объектах из U с весом w при y=1 и 1-w при y=0ю Возвращает None"""
        self.model= model
        self.estimator_type = estimator_type
        self.const = []
        self.cv_score = []
        self.weighted = weighted

    def choose_random_s(self, x, y):
        chosen = np.random.randint(0,2, size = y.shape)
        labels = y[(y==1) & (chosen==1)]
        x_labeled = x[(y==1)&(chosen==1)]
        x_unlabeled = x[(y==0)|(chosen==0)]
        y_no_label = y[(y==0)|(chosen==0)]
        no_label = np.zeros(y_no_label.shape[0])
        x=np.concatenate((x_unlabeled, x_labeled), axis = 0)
        y = np.concatenate((y_no_label, labels), axis=0)[:, np.newaxis]
        s = np.concatenate((no_label, labels), axis=0)[:, np.newaxis]
        ys = np.concatenate((y, s), axis=1)
        return x, ys

    def train(self, X_train, ys_train, X_val, ys_val):
        self.model.fit(X_train, ys_train[:,1])
        kf = KFold(n_splits=3)
        for train, test in kf.split(X_val):
            g =self.model.predict_proba(X_val)[train,1]
            c = self.estimator(g, ys_val[train,1])
            self.const.append(c)
            self.cv_score.append(self.test(X_val[test], ys_val[test, 0], c)[0])
        const = self.const[self.cv_score.index(max(self.cv_score))]
        if self.weighted:
            g =self.model.predict_proba(X_val)[:,1]
            y = np.concatenate((np.zeros(ys_val.shape[0]), np.ones(ys_val.shape[0])))
            weights = self.weight(g, const, ys_val[:,1])
            y[weights==1]=1
            self.model.fit(np.concatenate((X_val, X_val), axis = 0), y, sample_weight=weights)
        else:
            return const

    def test(self, X_test, y_test, const):
        proba = self.model.predict_proba(X_test)[:,1]/const
        k=[]
        k.append(roc_auc_score(y_test, proba))
        k.append(accuracy_score(y_test, np.around(proba)))
        return k

    def estimator(self, proba, s):
        if self.estimator_type==1:
            c = np.sum(proba*s)/s[s==1].shape[0]
            return c
        elif self.estimator_type==2:
            return np.sum(proba*s)/np.sum(proba)
        else:
            return np.max(proba)

    def weight(self, g, constant, s):
        w1 = ((1-constant)/constant)*(g/(np.ones(g.shape[0])-g))
        w1 = np.nan_to_num(w1)
        w1[s==1] = 1
        w2 = np.ones(w1.shape[0]) - w1
        return np.concatenate((w1, w2)) 
