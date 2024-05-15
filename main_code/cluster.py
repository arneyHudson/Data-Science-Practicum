import sklearn as sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
from abc import ABCMeta, abstractmethod
from scipy.stats import chi2_contingency, kstest
import numpy as np
from functools import reduce
from itertools import chain
from tqdm import tqdm
from multiprocessing import Process, Queue
import pandas as pd

# Add your pipelines here so that others can use them easily
pipelines = {
    'standard': Pipeline([('scaler', StandardScaler())])
}

class Cluster(metaclass=ABCMeta):

    def __init__(self, data):
        self._data = data
        self.results = None
        self.model = None
        self.labels = []

    @abstractmethod
    def fit(self, features, pipeline, cpus=2):
        raise NotImplementedError

    def get_cluster(self, cluster_id):
        return self._data[self._data['clusters'] == cluster_id]

    def get_exclude_cluster(self, cluster_id):
        return self._data[self._data['clusters'] != cluster_id]


    @classmethod
    def homogeneity(df):
        return metrics.homogeneity_score()

    def chi2(self, var1, cluster_id):
        combination_counts = pd.crosstab(self._data[var1], self._data['clusters'] == cluster_id)
        return chi2_contingency(combination_counts)

    def kstest(self, var, cluster_id):
        """
        Null Hypothesis: The two distributions are identical
        Alternate: The two distributions are different.
        """
        return kstest(self.get_cluster(cluster_id)[var], self.get_exclude_cluster(cluster_id)[var])

    def silhouette_score(self, var):
        return metrics.silhouette_score(np.array(self._data[var]).reshape(-1, 1), self.labels)

    def contingency_matrix(self, clustering):
        return metrics.cluster.contingency_matrix(clustering, self._data['clusters'])

class DBSCAN_Base(Cluster):

    def fit(self, features, pipeline, cpus):
        features = pipeline.fit(self._data[features]).transform(self._data[features])

        self.model = sklearn.cluster.DBSCAN(eps=2, n_jobs=cpus).fit(features)
        self._data['clusters'] = self.model.labels_
        self.labels = self.model.labels_


class FastCluster(Cluster):
    """
    Splits the dataset into a specified number of segments and creates a cluster on each segment then combines the labels
    """

    def __init__(self, data, cluster_obj):
        super().__init__(data)
        self.cluster_obj = cluster_obj

    def fit(self, features, pipeline, cpus, splits):
        models = []

        split_data = np.array_split(self._data, splits)

        for df in split_data:
            models.append(self.cluster_obj(df))

        queue = Queue()
        processes = []

        def _fit(model, features, pipeline, cpus, queue):
            model.fit(features, pipeline, cpus)
            queue.put(model.model.labels_)

        for m in models:
            if cpus > 0:
                p = Process(target=_fit, args=(m, features, pipeline, 2, queue))
                p.start()
            else:
                _fit(m, features, pipeline, 2, queue)
            cpus -= 1
            processes.append(p)

        labels = []
        for q in tqdm(range(len(models))):
            labels.append(queue.get())


        for p in processes:
            p.join()


        self._data['clusters'] = list(chain(*labels))
        self.labels = list(chain(*labels))
