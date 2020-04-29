import numpy as np
import scipy.spatial.distance as dist
from sklearn.datasets import load_digits
from collections import Counter

def calc_MI(x, y, bins=8):
    c_xy = np.histogram2d(x, y, bins)[0]
    n, m = c_xy.shape
    row_flags = [True]*n
    col_flags = [True]*m
    for i in range(n):
        if sum(c_xy[i]) == 0:
            row_flags[i] = False
        if sum(c_xy[:,i]) == 0:
            col_flags[i] = False
    c_xy = c_xy[row_flags,:]
    c_xy = c_xy[:, col_flags]
    
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi/np.log(2)

def calc_HX(x, bins=8):
    p = np.histogram(x, bins,density=True)[0]
    s = np.sum(p)
    P = [item/s for item in p]
    Hx = np.sum([-item*np.log2(item) for item in P if item != 0])
    return Hx

def calc_classHX(label):
    s = len(label)
    print(s)
    Hx = 0
    for c, p in Counter(label).most_common():
        Hx += -(p/s)*np.log2(p/s)
    return Hx


class EGSG:
    def __init__(self):
        self._scores = None
        self._indexs = None
        self._classHX = 0
        self._weight = 0
        self._t = 0
        self._k = 0
        self._bins = bins
        self._feature_groups = []


    def fit(self, data, label):
        self._classHX = calc_classHX(label)
        # step1
        ranks, scores = self._get_feature_ranks(data, label)
        
        # step2
        self._get_feature_groups(data, ranks, scores)

        # step3
        self._indexs = self._select_core_features(self._feature_groups, self._t)

    def fit_transform(self, data, label):
        pass

    def transform(self, data):
        pass
    
    # step1
    def _get_feature_ranks(self, data, label):
        scores = []
        for feature in data.T:
            MI = calc_MI(feature, label)
            hx = calc_HX(feature)
            ICC = MI/(hx + self._classHx - MI)
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
                mi = calc_MI(data[:,core_rank], data[:, feature_rank])
                hx = calc_HX(data[:, core_rank])
                hy = calc_HX(data[:, feature_rank])
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


if __name__ == '__main__':
    print(2333)