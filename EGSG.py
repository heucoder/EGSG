import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from collections import Counter


def cut(data, bins = 8):
    return pd.cut(data, bins, labels=range(bins)).codes

def calc_MI(x, y, x_bins=8, y_bins=8, x_discretization=False, y_discretization=False):
    if x_discretization:
        x = cut(x, x_bins)
    if y_discretization:
        y = cut(y, y_bins)
    return mutual_info_score(x, y)

def calc_HX(x, bins = 8, discretization=False):
    if discretization:
        x = cut(x, 8)
    s = len(x)
    Hx = 0
    for c, p in Counter(x).most_common():
        Hx += -(p/s)*np.log2(p/s)
    return Hx

class EGSG:
    def __init__(self, attr, t=1, *,label_discretization = False, label_bins = 8, x_discretization=False, x_bins=8, weight = 1.0):
        self._scores = None
        self._indexs = None

        self._weight = weight
        self._t = 0
        self._k = attr

        self._feature_groups = []

        self._label_discretization = label_discretization
        self._label_bins = label_bins
        self._x_discretization = x_discretization
        self._x_bins = x_bins

    def fit(self, data, label):
        # step1
        print("step1")
        ranks, scores = self._get_feature_ranks(data, label)
        print(ranks)
        # step2
        print("step2")
        self._get_feature_groups(data, ranks, scores)

        # step3
        print("step3")
        self._indexs = np.array(self._select_core_features(self._feature_groups, self._t))
        self._scores = np.abs(np.array(scores)[self._indexs])

    def fit_transform(self, data, label):
        self.fit(data, label)
        return data[:, self._indexs]

    def transform(self, data):
        if self._indexs:
            return data[:, self._indexs]
        else:
            pass
    
    # step1
    def _get_feature_ranks(self, data, label):
        scores = []
        classHX = calc_HX(label, self._label_bins, self._label_discretization)
        for feature in data.T:
            MI = calc_MI(feature, label)
            hx = calc_HX(feature)
            ICC = MI/(hx + classHX - MI)
            scores.append(-ICC)

        return np.argsort(scores)[:self._k], scores


    # step2
    def _get_feature_groups(self, data, feature_ranks, feature_scores):
        self._feature_groups = []
        self._feature_groups.append([feature_ranks[0]])
        for feature_rank in feature_ranks[1:]:
            flag = 1
            for feature_group in self._feature_groups:
                core_rank = feature_group[0]
                mi = calc_MI(data[:,core_rank], data[:, feature_rank], self._x_bins, self._x_discretization, self._x_bins, self._x_discretization)
                hx = calc_HX(data[:, core_rank], self._x_bins, self._x_discretization)
                hy = calc_HX(data[:, feature_rank], self._x_bins, self._x_discretization)
                if mi/(hx+hy-mi) < self._weight*feature_scores[feature_rank]:
                    feature_group.append(feature_rank)
                    flag = 0
                    break
            
            if flag:
                self._feature_groups.append([feature_rank])


    # step3
    def _select_core_features(self, feature_groups, t = 10):
        core_feature_ranks = []
        for feature_group in feature_groups:
            core_feature_ranks.append(feature_group[0])
            if len(feature_group) > 1:
                random_rank = np.random.choice(feature_group[1:t])
                core_feature_ranks.append(random_rank)
        
        return core_feature_ranks
            


def test():
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    X, y = load_digits(return_X_y=True)
    LR = LogisticRegression()
    egsg = EGSG(40, x_discretization=10, x_bins=10)
    X_new = egsg.fit_transform(X, y)
    print(X.shape, X_new.shape)


if __name__ == '__main__':
    test()