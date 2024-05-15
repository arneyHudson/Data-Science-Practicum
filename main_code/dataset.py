import pandas as pd
import re
from sklearn.impute import KNNImputer
from scipy import stats
import os

class DataSet:
    """
    Loads the CSV files in as dataframes in a cleaned and merged format.
    Anything outside of this should be placed in a different file 
    to avoid having to reload the dataset whenever this file is changed.
    """

    def __init__(self, path="/data/csc4801/ScotForge_Project1/012624/"):
        """
        Load and process dataframes
        """
        self._address_df = pd.read_csv(path + "Addresses.csv")
        self._grades_df = pd.read_csv(path + "Grades.csv")
        self._customers_df = pd.read_csv(path + "Customers.csv")
        self._nc_df = pd.read_csv(path + "NC.csv")
        self._sales_orders_df = pd.read_csv(path + "Sales Orders.csv")
        self._work_orders_df = pd.read_csv(path + "Work Orders.csv")

        del self._sales_orders_df["Prepaid Material Flag"]

        self.__clean_addr()
        self.__clean_customers()
        self.__impute_work_orders()

        self.__convert_grade_types()
        self.__convert_customer_types()
        self.__convert_sales_types()

        self._merged = self.__merge_into_sales_orders()
        self._gnw = self.__merge_into_gnw()
        
        self._final_customer_df = self.__all_into_customers()
        file_path = os.path.join(os.path.dirname(__file__), '../csv_datasets/')
        customer_location_df = pd.read_csv(file_path + "Customer_Location.csv")

        customer_location_df = customer_location_df[customer_location_df['Customer ID'].isin(self._final_customer_df['Customer ID'])]
        customer_location_df.reset_index(drop=True, inplace=True)

        # Add latitude and longitude columns into 2.5k dataset
        self._final_customer_df = pd.merge(self._final_customer_df, customer_location_df[['Customer ID', 'Latitude', 'Longitude']], on='Customer ID', how='left')

    @property
    def address_df(self):
        return self._address_df
    
    @property
    def grades_df(self):
        return self._grades_df
    
    @property
    def customers_df(self):
        return self._customers_df
    
    @property 
    def nc_df(self):
        return self._nc_df
    
    @property
    def sales_orders_df(self):
        return self._sales_orders_df
    
    @property
    def work_orders_df(self):
        return self._work_orders_df

    @property
    def merged_df(self):
        """
        All dataframes are merged into a copy of sales orders
        """
        return self._merged
    
    @property
    def gnw(self):
        return self._gnw
    
    @property
    def final_customer_df(self):
        return self._final_customer_df
    
    
    def get_empty_customer_values(self):
        return self._customers_df.isnull().sum()
    
    def __clean_addr(self):

        self.filtered_df = self._address_df[~self._address_df["Country"].isin(["US", "CA"])]

        def is_normal_address(address):
            """
            This function will look for if the address contains a number and then letters for the street name
            We want to eliminate any non-normal address as they will likely not be as helpful
            """
            if isinstance(address, str):
                # Regular expression pattern to match a number followed by a street name
                pattern = r"\d+\s+\S+.*"
                
                # Search for the pattern in the address
                match = re.search(pattern, address)
                
                return bool(match)
            
        address_df_copy = self._address_df.copy()

        # List to store indices of rows to be deleted
        rows_to_delete = []

        # Iterate over each row
        for idx, row in address_df_copy.iterrows():
            # Iterate over each address line column
            for i in range(1, 5):
                address_col = f"Address Line {i}"
                address = row[address_col].strip() if isinstance(row[address_col], str) else None
                if address and is_normal_address(address):
                    # Address is valid, move to the next row
                    break
                elif i < 4:  # Check if there's another address line to replace this one
                    next_address_col = f"Address Line {i+1}"
                    next_address = row[next_address_col].strip() if isinstance(row[next_address_col], str) else None
                    if next_address and is_normal_address(next_address):
                        # Replace the current address with the next one
                        row[address_col] = next_address
                        # Remove the next address
                        row[next_address_col] = None
                        break
            else:
                # If none of the addresses are valid, mark the row for deletion
                rows_to_delete.append(idx)
        
        # Drop rows that are marked for deletion
        address_df_copy.drop(index=rows_to_delete, inplace=True)

        self._address_df = address_df_copy

        self._address_df.reset_index(drop=True, inplace=True)

        zc_rep = []

        for zc in self._address_df["Zip Code"]:
            if "-" in zc:
                zc_rep.append(zc.split("-")[0])
            elif len(zc) == 9:
                zc_rep.append(zc[:5])  # truncate to the first 5 characters
            else:
                zc_rep.append(zc)

        self._address_df["Zip Code"] = zc_rep

        self._address_df.drop(columns=['Address Line 2', 'Address Line 3', 'Address Line 4'], inplace=True)

        zc_rep = []

        for zc in self._customers_df["Zip Code"]:
            str_zc = str(zc)
            if isinstance(zc, str) and "-" in zc:
                zc_rep.append(zc.split("-")[0])
            elif len(str_zc) == 9:
                zc_rep.append(zc[:5])  # truncate to the first 5 characters
            else:
                zc_rep.append(zc)

        self._customers_df["Zip Code"] = zc_rep

    def __clean_customers(self):
        cols_to_drop = ['Secondary Address', 'PO Box'] # These aren't very helpful in the grand scheme because there are so many empty values
        self._customers_df.drop(columns=cols_to_drop, inplace=True)
        
        transit_columns = ["Transit Days from Ringmasters", "Transit Days from Clinton", "Transit Days from Spring Grove", "Transit Days from Franklin Park"]
        self._customers_df["International"] = self._customers_df["Country Code"].apply(lambda x: 1 if x not in ["US", "CA"] else 0)
        self._customers_df.loc[self._customers_df["International"] == 1, transit_columns] = "International" # Changes the null values for International Shipments

        # Manually Filling in small amounts of N/A values
        self._customers_df.at[261, "City"] = "Singapore"

        self._customers_df.at[7018, "City"] = "Ecclesfield"
        self._customers_df.at[7018, "Region Code"] = "UKENG"
        self._customers_df.at[7018, "Zip Code"] = "S35 9TG"
        
        empty_region_code_rows = pd.DataFrame(self._customers_df[self._customers_df["Region Code"].isnull()])
        empty_region_code_rows.head()


        # Marly, Switzerland
        self._customers_df.at[2662, "Zip Code"] = "1723"
        self._customers_df.at[2662, "Region Code"] = "CHFR" # Marly is within the Canton (US state equivalent) of Fribourg



        # Dubai Customer
        self._customers_df.at[7226, "Region Code"] = "UAEDB"
        self._customers_df.at[7226, "Zip Code"] = "00000"

    
    def __convert_grade_types(self):
        self._grades_df["Grade"] = pd.Categorical(self._grades_df["Grade"])
        self._grades_df["Grade Family"] = pd.Categorical(self._grades_df["Grade Family"])

    def __convert_customer_types(self):
        categories = ["Customer Status", "Customer Type", "Pricing Category", "Country Code", "Default Currency Indicator",
              "Customer Category", "Region Code", "Account Manager", "Transit Days from Ringmasters", "Transit Days from Clinton", "Transit Days from Franklin Park",
              "Transit Days from Spring Grove", "Salesperson", "Credit Limit"]

        for c in categories:
            self._customers_df[c] = pd.Categorical(self._customers_df[c])

        texts = ["Zip Code", "City", "Prospect Date Opened", "Customer Date Opened", "Customer Last Activated Date", "Customer Name", "Primary Address"]

        for t in texts:
            self._customers_df[t] = self._customers_df[t].astype('string')

        self._customers_df["Customer ID"] = self._customers_df["Customer ID"].astype('int32')

    def __convert_sales_types(self):
        self._sales_orders_df["Shipping Method"] = pd.Categorical(self._sales_orders_df["Shipping Method"])

        texts = ["Sold To Name", "Ship To Name", "Ship Date"]

        for t in texts:
            self._sales_orders_df[t] = self._sales_orders_df[t].astype('string')

        int_32 = ["Sales Order Number", "Work Order Number", "Sold To ID", "Ship To ID"]

        for i in int_32:
            self._sales_orders_df[i] = self._sales_orders_df[i].astype("int32")

    def __convert_sales_types(self):
        texts = ["Customer Name", "Release Plant", "Order Date", "Ship Date"]
        categories = ["Shape Code", "Order Status", "Grade"]

        for t in texts:
            self._work_orders_df[t] = self._work_orders_df[t].astype('string')
            
        for c in categories:
            self._work_orders_df[c] = pd.Categorical(self._work_orders_df[c])

    def __impute_work_orders(self):
        X = self._work_orders_df[["Total Shipped Quantity"]]

        knn_imputer = KNNImputer(n_neighbors=10)

        self._work_orders_df["Total Shipped Quantity"] = knn_imputer.fit_transform(self._work_orders_df[["Total Shipped Quantity"]])

        self._work_orders_df["Total Shipped Quantity"] = self._work_orders_df["Total Shipped Quantity"].astype(int)
            
    def __merge_into_sales_orders(self):
        gnw = self.__merge_into_gnw()
        cgnw = gnw.merge(self._customers_df, on="Customer ID", how="left")
        self._sales_orders_df.set_index("Work Order Number")
        return self._sales_orders_df.merge(cgnw, how="left")
    
    def __merge_into_gnw(self):
        self._work_orders_df.set_index("Work Order Number")
        defect_count = self._nc_df.value_counts(self._nc_df["Work Order Number"])
        nc_work = self._work_orders_df.merge(defect_count.rename("defects"), on="Work Order Number", how="left")
        nc_work.defects.mask(pd.isna(nc_work.defects), other=0, inplace=True)
        #This line of code will sum up the NC Serial Count (number of non compliant parts) and the MRR Serial Count (number NC due to material rejection) based on the work order.
        nc_sum = self.nc_df[['Work Order Number','NC Serial Count','MRR Serial Count']].groupby('Work Order Number').sum() # added by ian
        #This will drop the null rows (rows where the total shipped quantity is NaN because they were not shipped)
        nc_work = nc_work.dropna().merge(nc_sum.reset_index(),on='Work Order Number',how='left') # added by ian
        #This will fill in the zeros from work orders where there was no NC or MRR
        nc_work['NC Serial Count'] = nc_work['NC Serial Count'].fillna(0) # added by ian
        nc_work['MRR Serial Count'] = nc_work['MRR Serial Count'].fillna(0) # added by ian
        gnw = nc_work.merge(self._grades_df, on="Grade", how="left")
        return gnw
    
    def __all_into_customers(self):
        gnw = self.__merge_into_gnw()
        gnw = gnw.dropna()
        so = self._sales_orders_df
        so.set_index("Work Order Number")
        gnws = so.merge(gnw, how="left")
        gnws = gnws.dropna()
        gnws = gnws.drop(columns = ["Sales Order Number", "Work Order Number", "Sold To ID", "Sold To Name", "Ship To ID", 
                        "Ship To Name", "Ship Date"])
        agg = gnws.groupby('Customer ID').agg({
            'Line Number': lambda x: stats.mode(x)[0][0],
            'Shipping Method': lambda x: stats.mode(x)[0][0], 
            'Customer Name': lambda x: stats.mode(x)[0][0],
            'Release Plant': lambda x: stats.mode(x)[0][0],
            'Shape Code': lambda x: stats.mode(x)[0][0],
            'Order Date': lambda x: stats.mode(x)[0][0],
            'Order Status': lambda x: stats.mode(x)[0][0],
            'Total Order Weight': 'mean',
            'Total Order Price': 'mean',
            'Total Shipped Quantity': 'median',
            'Grade': lambda x: stats.mode(x)[0][0],
            'defects': lambda x: stats.mode(x)[0][0],
            'NC Serial Count': 'median',
            'MRR Serial Count': 'median',
            'Grade Family': lambda x: stats.mode(x)[0][0],
            'Material Density': 'mean'
        }).reset_index()
        cust = self.customers_df
        final = cust.merge(agg, on='Customer ID', how="left")
        final = final.dropna()
        return final

