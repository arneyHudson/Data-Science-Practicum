import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# from mod.visualization import Visualize
class Visualize():
    def graph(self, clusters):
        """
        clusters: an array of the labeles for the data
        """
        sns.countplot(clusters)
        plt.title("Cluster Distribution")
