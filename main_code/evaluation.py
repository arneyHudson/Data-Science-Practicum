import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, kstest

def check_cell_count(df, feature, threshold=5, alpha=0.2):
    """
    Calculate whether the row of a contingency table meets the minimum cell count assumption 
    for the pearson's chi2 test for homogenity.
    Learn more at https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Assumptions

    threshold: the minimum count a category should have. See alpha definition.
    alpha: At what percentage of categories have less than the count. 
    """
    return len(df[df[feature].map(df[feature].value_counts()) < threshold][feature].value_counts())/len(df) < alpha

def plot_test_statistic(feature, res, test_name, pvalue_thresh=0.05):
    """
    Generates a plot of the kstest test statistic for each cluster ID 
    that has a significantly different distribution than the rest of the data for the entered feature.
    """
    d = list(map(lambda r: (r, res[feature][r][0], res[feature][r][1]), res[feature]))
    sales = pd.DataFrame(d)
    #Only show statistically significant results
    sales = sales[sales[2] < pvalue_thresh]
    ax = sales[1].plot.bar()
    ax.set_xticklabels(sales[0])
    ax.set_title(feature)
    ax.set_ylabel(test_name + " Test Statistic")
    ax.set_xlabel("Cluster ID")

class Evaluation():
    def __init__(self, customers_df, manufacturing_df, labels):
        self.customers_df = customers_df
        self.manufacturing_df = manufacturing_df
        self.customers_df['clusters'] = labels        
        self.clusters = customers_df['clusters'].unique()

    def plot_distribution(self):
        ax = self.customers_df['clusters'].value_counts().plot.bar()
        ax.set_title("Cluster Distribution")
        ax.set_ylabel("Number of Customers in Cluster")
        ax.set_xlabel("Cluster ID (-1 is miscellaneous unclustered data)")
    
    
    def split_data(self, cluster_id):
        """
        Returns all customers with the given cluster_id and in another dataframe everything outside that cluster.
        """
        clust = self.manufacturing_df[self.manufacturing_df['Customer ID'].
                                      isin(self.customers_df[self.customers_df['clusters'] == cluster_id]['Customer ID'])]
        outer = self.manufacturing_df[self.manufacturing_df['Customer ID'].
                                      isin(self.customers_df[self.customers_df['clusters'] != cluster_id]['Customer ID'])]
        return clust, outer

    def __compare_clusters(self, features, test):
        res_categorical = {}
        for c in features:
            res_categorical[c] = {}
            print(c)
            for cluster_id in self.clusters:
                clust, outer = self.split_data(cluster_id)

                clust['clusters'] = True
                outer['clusters'] = False


                if (len(clust) != 0 and len(outer) != 0):
                    res_categorical[c][cluster_id] = test(clust, outer, c)
                else:
                    res_categorical[c][cluster_id] = np.nan

        return res_categorical
        
    def compare_clusters_categorical(self):

        # CHI2 Homogenity Tests for Categorical
        # For many categorical variables it makes sense to use pearson's chi-squared test for homogenity
        # Assumptions: 
        # - Each "cell" has at least 5 or 10 samples. A cell is a pairing between a category and a group.
        #   For example if category Apple in cluster 1 only has 2 samples, we would not meet the assumption
        # - A large sample size: For many clusters this will hold true, but for some this won't
        # - Independence: We can assume each cluster is independent as each transaction or customer should be independent. 
        #   Is this completely true? Perhaps not, but is still a safe and mostly true assumption.
        
        categorical = [
            'Shipping Method',
            'Shape Code',
            'Order Status',
            'Grade',
            'Grade Family',
            'Customer Status',
            'Customer Type',
            'SIC Code',
            'Default Currency Indicator',
            'Customer Category',
            'International',
            'Transit Days from Franklin Park', 
            'Transit Days from Spring Grove', 
            'Transit Days from Clinton', 
            'Transit Days from Ringmasters', 
            'Transit Days from Ringmasters'
        ]

        def chi2(clust, outer, c):
            combined = pd.concat([clust, outer])
            return chi2_contingency(pd.crosstab(combined[c], combined['clusters']))

        return self.__compare_clusters(categorical, chi2)

    def compare_clusters_ordinal(self):
        
        ordinal = [
            'Credit Limit', 'Salesperson', 
            'Quote Speed',
            'Customer Last Activated Date',
            'Customer Date Opened',
            'Prospect Date Opened',
            'Material Density',
            'MRR Serial Count',
            'NC Serial Count',
            'Total Shipped Quantity',
            'Total Order Price',
            'Total Order Weight',
            'Order Date',
        ]

        test = lambda clust, outer, c: kstest(clust[c].dropna(), outer[c].dropna())
        return self.__compare_clusters(ordinal, test)