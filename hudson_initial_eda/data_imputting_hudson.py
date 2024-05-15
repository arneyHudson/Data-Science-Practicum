# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data Cleaning and Imputting
# - Principal Author: Hudson Arney
# - Date: 2/19/2024

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re
from sklearn.impute import KNNImputer
import numpy as np
import requests
from tqdm import tqdm
import folium
import plotly.graph_objects as go
from json.decoder import JSONDecodeError

# %%
# Show all columns
pd.set_option('display.max_columns', None)

# %% [markdown]
# The Prepaid Material Flag has the same value on every entry, so is dropped.

# %%
path = "/data/csc4801/ScotForge_Project1/012624/"

#address_df = pd.read_csv(path + "Addresses.csv")
address_df = pd.read_csv("Geocoded_Addresses.csv")
grades_df = pd.read_csv(path + "Grades.csv")
customers_df = pd.read_csv(path + "Customers.csv")
nc_df = pd.read_csv(path + "NC.csv")
sales_orders_df = pd.read_csv(path + "Sales Orders.csv")
work_orders_df = pd.read_csv(path + "Work Orders.csv")

del sales_orders_df["Prepaid Material Flag"]

# %% [markdown]
# ## Address Data Table Cleaning

# %%
address_df.tail()

# %%
empty_values = address_df.isnull().sum()
print(empty_values)

# %%
# City and Country are important for looking at longitude and latitude, address will just make it more precise
columns_to_replace = ['City', 'Country']
address_df[columns_to_replace] = address_df[columns_to_replace].replace(r'^\s*$', np.nan, regex=True)
address_df.dropna(subset=columns_to_replace, how='all', inplace=True)

address_df.reset_index(drop=True, inplace=True)

# %%
#def is_normal_address(address, country):
    #if isinstance(address, str):
        #if country in ["US", "CA"]:
            #pattern = r"\d+\s+\S+.*"  # Pattern for US and Canada addresses
            #match = re.search(pattern, address)
            #return bool(match)

# %%
#address_df_copy = address_df.copy()

# List to store indices of rows to be deleted
#rows_to_delete = []

# Iterate over each row
#for idx, row in address_df_copy.iterrows():
    # Iterate over each address line column
    #for i in range(1, 5):
        #address_col = f"Address Line {i}"
        #address = row[address_col].strip() if isinstance(row[address_col], str) else None
        #if address and is_normal_address(address, row["Country"]):
            #break
        #elif i < 4:  # Check if there's another address line to replace this one
            #next_address_col = f"Address Line {i+1}"
            #next_address = row[next_address_col].strip() if isinstance(row[next_address_col], str) else None
            #if next_address and is_normal_address(next_address, row["Country"]):
                # Replace the current address with the next one
                #row[address_col] = next_address
                # Remove the next address
                #row[next_address_col] = None
                #break
    #else:
        # If none of the addresses are valid, mark the row for deletion
        #rows_to_delete.append(idx)

# Drop rows that are marked for deletion
#address_df_copy.drop(index=rows_to_delete, inplace=True)

# Print the deleted rows
#print("Deleted rows:")
#print(address_df.loc[rows_to_delete])

#address_df = address_df_copy

#address_df.reset_index(drop=True, inplace=True)

# %%
zc_rep = []

for zc in address_df["Zip Code"]:
    if "-" in zc:
        zc_rep.append(zc.split("-")[0])
    elif len(zc) == 9:
        zc_rep.append(zc[:5])  # truncate to the first 5 characters
    else:
        zc_rep.append(zc)

address_df["Zip Code"] = zc_rep


# %%
#address_df.drop(columns=['Address Line 2', 'Address Line 3', 'Address Line 4'], inplace=True)
#address_df.head(-1)

# %%
def get_geocode(address, city, state, country):
    latitude, longitude = None, None
    base_url = 'https://nominatim.openstreetmap.org/search'
    query = ''
    if address:
        query += address
    if city:
        query += f", {city}"
    if state:
        query += f", {state}"
    if country:
        query += f", {country}"
    
    if not query:
        return None, None
    
    # Make the request to the API
    
    params = {
        'q': query,
        'format': 'json'
    }
    response = requests.get(base_url, params=params)
    try:
        data = response.json()
        
        if data:
            latitude = float(data[0]['lat'])
            longitude = float(data[0]['lon'])
            return latitude, longitude
    except JSONDecodeError as e:
        print("Error decoding JSON response:", e)
        print("Response content:", response.content)
    
    
    return latitude, longitude

# %% [markdown]
# ____________________________________________________________________________________________________________________

# %%
# Trying to get the Geocode location data using openstreetmap api

#for index, row in tqdm(address_df.iterrows(), total=len(address_df)):
    #address = row['Address Line 1']
    #city = row['City']
    #state = row['State']
    #country = row['Country']
    #latitude, longitude = get_geocode(address, city, state, country)
    #address_df.at[index, 'Latitude'] = latitude
    #address_df.at[index, 'Longitude'] = longitude


# %%
# Save DataFrame to CSV with geocoded coordinates
# address_df.to_csv('Geocoded_Addresses.csv', index=False)

# %% [markdown]
# ____________________________________________________________________________________________________________________

# %% [markdown]
# ## HTML Map that can be found in the directory. More detail with the expense of performance.

# %%
# Create a base map
m = folium.Map(location=[0, 0], zoom_start=3)

# Filter out rows with non-null latitude and longitude
valid_rows = address_df[~address_df[['Latitude', 'Longitude']].isnull().any(axis=1)]

# Iterate over valid rows and add markers for each coordinate
for index, row in valid_rows.iterrows():
    latitude = row['Latitude']
    longitude = row['Longitude']
    folium.Marker([latitude, longitude]).add_to(m)

# Save the map to an HTML file
m.save('map.html')


# %%
address_customer_df = address_df.merge(customers_df[['Customer ID', 'Customer Name']], on='Customer ID', how='left')

# %% [markdown]
# __________________________________________________________________________________________________________________________

# %%
columns_to_replace_2 = ['Address Line 2']
address_df[columns_to_replace_2] = address_df[columns_to_replace_2].replace(r'^\s*$', np.nan, regex=True)

columns_to_replace_3 = ['Address Line 3']
address_df[columns_to_replace_3] = address_df[columns_to_replace_3].replace(r'^\s*$', np.nan, regex=True)

# Drop rows with NaN values in the specified columns
non_empty_address_2_df = address_df.dropna(subset=columns_to_replace_2, how='all')
non_empty_address_3_df = address_df.dropna(subset=columns_to_replace_3, how='all')

# Reset index
non_empty_address_2_df.reset_index(drop=True, inplace=True)
non_empty_address_3_df.reset_index(drop=True, inplace=True)

# Print the number of rows
print("Number of rows in non_empty_address_2_df:", len(non_empty_address_2_df))
print("Number of rows in non_empty_address_3_df:", len(non_empty_address_3_df))

# %%
non_empty_address_2_df = pd.read_csv('Addresses_2.csv')

# %%
non_empty_address_3_df = pd.read_csv('Addresses_3.csv')

# %% [markdown]
# _______________________________________________________________________________________

# %%
address_customer_2_df = non_empty_address_2_df.merge(customers_df[['Customer ID', 'Customer Name']], on='Customer ID', how='left')
address_customer_3_df = non_empty_address_3_df.merge(customers_df[['Customer ID', 'Customer Name']], on='Customer ID', how='left')

# %% [markdown]
# ## In Notebook Map that has good performance

# %%
fig = go.Figure()

# Add Companies 1st Address 1 (green)
fig.add_trace(go.Scattergeo(
    lon=address_customer_df['Longitude'],
    lat=address_customer_df['Latitude'],
    mode='markers',
    marker=dict(color='darkgreen', size=4),
    hoverinfo='text',  # Set hoverinfo to display text on hover
    text=address_customer_df['Customer Name'],  # Use customer name as hover text
    name='Address 1'  # Add name for the legend
))

# Add Companies for Address 2 (yellow)
fig.add_trace(go.Scattergeo(
    lon=address_customer_2_df['Longitude'],
    lat=address_customer_2_df['Latitude'],
    mode='markers',
    marker=dict(color='gold', size=4),
    hoverinfo='text',  # Set hoverinfo to display text on hover
    text=address_customer_2_df['Customer Name'],  # Use customer name as hover text
    name='Address 2'  # Add name for the legend
))

# Add Companies Address 3 (pink)
fig.add_trace(go.Scattergeo(
    lon=address_customer_3_df['Longitude'],
    lat=address_customer_3_df['Latitude'],
    mode='markers',
    marker=dict(color='magenta', size=4),
    hoverinfo='text',  # Set hoverinfo to display text on hover
    text=address_customer_3_df['Customer Name'],  # Use customer name as hover text
    name='Address 3'
))


states_geojson = requests.get(
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_1_states_provinces_lines.geojson"
).json()

fig = fig.add_trace(
    go.Scattergeo(
        lat=[
            v
            for sub in [
                np.array(f["geometry"]["coordinates"])[:, 1].tolist() + [None]
                for f in states_geojson["features"]
            ]
            for v in sub
        ],
        lon=[
            v
            for sub in [
                np.array(f["geometry"]["coordinates"])[:, 0].tolist() + [None]
                for f in states_geojson["features"]
            ]
            for v in sub
        ],
        line_color="gray",
        line_width=1,
        mode="lines",
        showlegend=False,
    )
)

fig.update_geos(
    projection_type="natural earth",
    showcountries=True,
    countrycolor='darkgray',
    showsubunits=True,
    subunitcolor='Brown'
    #landcolor='white',  # Color of land
    #showocean=True,
    #oceancolor='lightblue',
    #showrivers=True,
    #showlakes=True,
    #showcoastlines=True,
    #rivercolor='#006994',
    #coastlinecolor='#191970',
    #lakecolor='#006994',
    #riverwidth=0.3
)
fig.show()


# %%
address_df['Address 2'] = address_df['Customer ID'].isin(address_customer_2_df['Customer ID']).astype(int)
address_df['Address 3'] = address_df['Customer ID'].isin(address_customer_3_df['Customer ID']).astype(int)

# %%
address_df.drop(columns=['Address Line 2', 'Address Line 3', 'Address Line 4'], inplace=True)
address_df.dropna(subset=['Longitude', 'Latitude'], inplace=True)

# %%
# Filling in missing values
address_df.loc[address_df['Customer ID'] == 514015, 'City'] = 'Niantic'
address_df.loc[address_df['Customer ID'] == 518096, 'City'] = 'Houston'
address_df.loc[address_df['Customer ID'] == 517439, 'City'] = 'Frostproof'
address_df.loc[address_df['Customer ID'] == 519098, 'City'] = 'Lees Summit'
address_df.loc[address_df['Customer ID'] == 501122, 'City'] = 'Marree'
address_df.loc[address_df['Customer ID'] == 520373, 'City'] = 'Houston'
address_df.loc[address_df['Customer ID'] == 522282, 'Country'] = 'US'
address_df.loc[address_df['Customer ID'] == 522740, 'City'] = 'Kapkot'
address_df.loc[address_df['Customer ID'] == 523012, 'Country'] = 'US'
address_df.loc[address_df['Customer ID'] == 509350, 'City'] = 'The Hague'
address_df.loc[address_df['Customer ID'] == 523459, 'Country'] = 'US'


# %%
empty_values = address_df.isnull().sum()
print(empty_values)

# %%
address_df["International"] = address_df["Country"].apply(lambda x: 1 if x not in ["US"] else 0)

# %%
address_df.sample(5)

# %%
duplicated_rows = address_df[(address_df['Longitude'].shift() == address_df['Longitude']) & 
                              (address_df['Latitude'].shift() == address_df['Latitude'])]

# %%
duplicated_rows.size

# %% [markdown]
# # Looking at Percentages in the Location Data
#
# ### Country Percentages

# %%
total_customers = len(address_df)
country_counts = address_df['Country'].value_counts()
for country, count in country_counts.items():
    percentage_customers = (count / total_customers) * 100
    print(f"Percentage of customers from {country}: {percentage_customers:.2f}%")

# %% [markdown]
# ### Continent Percentages

# %%
continent_mapping = {
    'US': 'North America',
    'CA': 'North America',
    'MX': 'North America',
    'UK': 'Europe',
    'NL': 'Europe',
    'AU': 'Australia',
    'CN': 'Asia',
    'NZ': 'Australia',
    'CO': 'South America',
    'DE': 'Europe',
    'IL': 'Asia',
    'IN': 'Asia',
    'BR': 'South America',
    'SK': 'Europe',
    'CR': 'North America',
    'VE': 'South America',
    'KR': 'Asia',
    'FI': 'Europe',
    'BE': 'Europe',
    'IT': 'Europe',
    'FR': 'Europe',
    'PR': 'North America',
    'AE': 'Asia',
    'EC': 'South America',
    'ES': 'Europe',
    'PH': 'Asia',
    'AR': 'South America',
    'CL': 'South America',
    'CH': 'Europe',
    'GT': 'North America',
    'AT': 'Europe',
    'PE': 'South America',
    'SG': 'Asia',
    'CZ': 'Europe'
}

total_customers = len(address_df)
continent_counts = address_df['Country'].map(continent_mapping).value_counts()

for continent, count in continent_counts.items():
    percentage_customers = (count / total_customers) * 100
    print(f"Percentage of customers from {continent}: {percentage_customers:.2f}%")

# %% [markdown]
# ### Breaking down subregions of the United States

# %%
west_states = ['CA', 'WA', 'OR', 'AK', 'HI', 'NV']
rocky_mountain_states = ['CO', 'WY', 'UT', 'ID', 'MT']
plains_states = ['ND', 'SD', 'NE', 'KS', 'IA', 'MO']
southwest_states = ['TX', 'AZ', 'NM', 'OK']
great_lakes_states = ['WI', 'MN', 'IL', 'MI', 'IN', 'OH']
mid_east_states = ['MD', 'DE', 'NJ', 'NY', 'PA']
new_england_states = ['CT', 'RI', 'MA', 'VT', 'NH', 'ME']
southeast_states = ['AL', 'AR', 'FL', 'GA', 'KY', 'LA', 'MS', 'NC', 'SC', 'TN', 'VA', 'WV']

# %%
state_to_region = {state: 'West' for state in west_states}
state_to_region.update({state: 'Rocky Mountain' for state in rocky_mountain_states})
state_to_region.update({state: 'Plains' for state in plains_states})
state_to_region.update({state: 'Southwest' for state in southwest_states})
state_to_region.update({state: 'Great Lakes' for state in great_lakes_states})
state_to_region.update({state: 'Mid East' for state in mid_east_states})
state_to_region.update({state: 'New England' for state in new_england_states})
state_to_region.update({state: 'Southeast' for state in southeast_states})

# %%
address_df['Region'] = address_df['State'].map(state_to_region)

fig = go.Figure(go.Choropleth(
    locations=address_df['State'],
    locationmode='USA-states',
    z=address_df['Region'].map({'West': 1, 'Rocky Mountain': 2, 'Plains': 3, 'Southwest': 4,
                                'Great Lakes': 5, 'Mid East': 6, 'New England': 7, 'Southeast': 8}),
    colorscale=[[0, 'yellow'], [0.14, 'gray'], [0.29, 'tan'], [0.43, 'orange'],
                [0.57, 'blue'], [0.71, 'purple'], [0.86, 'lightgreen'], [1, 'red']],
    showscale=False
))


# %%
#fig.update_layout(

#    title_text='Regions of the United States',
#    annotations=[
#        dict(
#            x=1.05,
#            y=i * 0.075,
#            xref="paper",
#            yref="paper",
#            text=f'<span style="color:black;">{region}</span> <span style="color:{color};">&#x25A0;</span>',
#            showarrow=False,
#            font=dict(
#                size=12,
#                color="black"
#            )
#        ) for i, (region, color) in enumerate([('West', 'yellow'), ('Rocky Mountain', 'gray'), 
#                                               ('Plains', 'tan'), ('Southwest', 'orange'), 
#                                               ('Great Lakes', 'blue'), ('Mid East', 'purple'), 
#                                               ('New England', 'lightgreen'), ('Southeast', 'red')])
#    ],
#    geo=dict(
#        scope='usa',
#        projection=go.layout.geo.Projection(type='albers usa'),
#        showland=True,
#        landcolor='rgb(217, 217, 217)',
#    )
#)'''

#fig.show()

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
def fill_null_regions(row):
    if pd.isnull(row['Region']):
        country = row['Country']
        if country in continent_mapping:
            return continent_mapping[country]
        else:
            return 'Other'
    return row['Region']

address_df['Region'] = address_df.apply(fill_null_regions, axis=1)

# %%
address_df['Region'] = address_df.apply(fill_null_regions, axis=1)

# Print percentage of customers from each region
total_customers = len(address_df)
region_counts = address_df['Region'].value_counts()

for region, count in region_counts.items():
    percentage_customers = (count / total_customers) * 100
    print(f"Percentage of customers from {region}: {percentage_customers:.2f}%")

# %%
empty_values = address_df.isnull().sum()

print(empty_values)

# %%
selected_columns = ['Customer ID', 'Country', 'Latitude', 'Longitude', 'Address 2', 'Address 3', 'Region']

# Create a new DataFrame with only the selected columns
new_df = address_df[selected_columns]

# Save the new DataFrame to a new CSV file
new_df.to_csv("Customer_Location.csv", index=False)

# %% [markdown]
# ________________________

# %% [markdown]
# _______________

# %% [markdown]
# ## Grade Data Table
# Nothing needed to be done

# %%
# grades_df.head(-1)
empty_values = grades_df.isnull().sum()

print(empty_values)

# %% [markdown]
# ## Customer Data Table
# Needed the most help. Most data that was International outside of Canada, was troubling.

# %%
num_unique_customer_ids = customers_df['Customer ID'].nunique()
print(num_unique_customer_ids)

# %%
customers_df.head(-1)

# %%

# %%
cols_to_drop = ['Secondary Address', 'PO Box'] # These aren't very helpful in the grand scheme because there are so many empty values
customers_df.drop(columns=cols_to_drop, inplace=True)

# %%
#empty_values = customers_df.isnull().sum()
#print(empty_values)

# %%
zc_rep = []

for zc in customers_df["Zip Code"]:
    str_zc = str(zc)
    if isinstance(zc, str) and "-" in zc:
        zc_rep.append(zc.split("-")[0])
    elif len(str_zc) == 9:
        zc_rep.append(zc[:5])  # truncate to the first 5 characters
    else:
        zc_rep.append(zc)

customers_df["Zip Code"] = zc_rep

# %%
# customers_df.head(-1)

# %%
transit_columns = ["Transit Days from Ringmasters", "Transit Days from Clinton", "Transit Days from Spring Grove", "Transit Days from Franklin Park"]
customers_df["International"] = customers_df["Country Code"].apply(lambda x: 1 if x not in ["US", "CA"] else 0)
customers_df.loc[customers_df["International"] == 1, transit_columns] = "International" # Changes the null values for International Shipments


# %%
# Manually Filling in small amounts of N/A values
customers_df.at[261, "City"] = "Singapore"

customers_df.at[7018, "City"] = "Ecclesfield"
customers_df.at[7018, "Region Code"] = "UKENG"
customers_df.at[7018, "Zip Code"] = "S35 9TG"

# %%
empty_values = customers_df.isnull().sum()
print(empty_values)

# %%
empty_region_code_rows = pd.DataFrame(customers_df[customers_df["Region Code"].isnull()])
empty_region_code_rows.head()


# Marly, Switzerland
customers_df.at[2662, "Zip Code"] = "1723"
customers_df.at[2662, "Region Code"] = "CHFR" # Marly is within the Canton (US state equivalent) of Fribourg

# Dubai Customer
customers_df.at[7226, "Region Code"] = "UAEDB"
customers_df.at[7226, "Zip Code"] = "00000"

# %%
empty_region_code = pd.DataFrame(customers_df[customers_df["City"] == "Marly"])
empty_region_code.head()

# %%
customers_df.head()

# %% [markdown]
# ## NC Data Table

# %%
# nc_df.head(-1)
empty_values = nc_df.isnull().sum()

print(empty_values)

# %% [markdown]
# ## Sales Data Table

# %%
# sales_orders_df.head(-1)
empty_values = sales_orders_df.isnull().sum()

print(empty_values)

# %% [markdown]
# ## Work Orders Data Table

# %%
# work_orders_df.head(-1)

empty_values = work_orders_df.isnull().sum()

print(empty_values)

# %%
work_orders_df.head(-1)

# %%
missing_shipping_rows = pd.DataFrame(work_orders_df[work_orders_df["Total Shipped Quantity"].isnull()])
missing_shipping_rows.head()

# %%
X = work_orders_df[["Total Shipped Quantity"]]

knn_imputer = KNNImputer(n_neighbors=10)

work_orders_df["Total Shipped Quantity"] = knn_imputer.fit_transform(work_orders_df[["Total Shipped Quantity"]])

work_orders_df["Total Shipped Quantity"] = work_orders_df["Total Shipped Quantity"].astype(int)

# %% [markdown]
# # Converting Data Types
#
# Without precision loss, most data types were converted to either string, Categorical, or an integer with a lower precision when possible.
# The remaining integer and floats could further be reduced in precision if fill in the NaN values.

# %%
grades_df["Grade"] = pd.Categorical(grades_df["Grade"])
grades_df["Grade Family"] = pd.Categorical(grades_df["Grade Family"])

# %%
categories = ["Customer Status", "Customer Type", "Pricing Category", "Country Code", "Default Currency Indicator",
              "Customer Category", "Region Code", "Account Manager", "Transit Days from Ringmasters", "Transit Days from Clinton", "Transit Days from Franklin Park",
              "Transit Days from Spring Grove", "Salesperson", "Credit Limit"]

for c in categories:
    customers[c] = pd.Categorical(customers[c])

texts = ["Zip Code", "City", "Prospect Date Opened", "Customer Date Opened", "Customer Last Activated Date", "Customer Name", "Primary Address", "Secondary Address"]

for t in texts:
    customers[t] = customers[t].astype('string')

customers["Customer ID"] = customers["Customer ID"].astype('int32')

# %%
sales_orders_df["Shipping Method"] = pd.Categorical(sales_orders_df["Shipping Method"])

texts = ["Sold To Name", "Ship To Name", "Ship Date"]

for t in texts:
    sales_orders_df[t] = sales_orders_df[t].astype('string')

int_32 = ["Sales Order Number", "Work Order Number", "Sold To ID", "Ship To ID"]

for i in int_32:
    sales_orders_df[i] = sales_orders_df[i].astype("int32")

# %%
texts = ["Customer Name", "Release Plant", "Order Date", "Ship Date"]
categories = ["Shape Code", "Order Status", "Grade"]

for t in texts:
    work_orders_df[t] = work_orders_df[t].astype('string')
    
for c in categories:
    work_orders_df[c] = pd.Categorical(work_orders_df[c])

# %% [markdown]
# # Merging the Data
#
# The sum of defects per work order is merged into a new dataframe. All other data from nc is discarded as I doubt it is very relevant and is limited in quantity. We can change this if neccessary. I'm highly skeptical that I did it correct considering there are a zillion rows and I barely know what I'm doing. **A further audit may be useful.**
#
# I did ensure that now rows were lost in the merge. As some data is missing there are many rows with "NaN" entries. Nearly all but 5 rows have these values. However each column has most of the values filled. An imputer is almost certainly neccessary if we want to use many of these dimensions.
#
# Addresses is not merged as it seems somewhat redundand and added a lot of NaN values. If we want this data we can merge it later.

# %%
work_orders_df.set_index("Work Order Number")
defect_count = nc_df.value_counts(nc_df["Work Order Number"])
nc_work = work_orders_df.merge(defect_count.rename("defects"), on="Work Order Number", how="left")
nc_work.defects.mask(pd.isna(nc_work.defects), other=0, inplace=True)
gnw = nc_work.merge(grades_df, on="Grade", how="left")
cgnw = gnw.merge(customers, on="Customer ID", how="left")
sales_orders_df.set_index("Work Order Number")
df = sales_orders_df.merge(cgnw, how="left")

df.info()

# %%
categorical_columns = []
numerical_columns = ["Total Order Weight", "Total Order Price", "Total Shipped Quantity", "defects", "Material Density", "Quote Speed", "Transit Days from Ringmasters", "Transit Days from Clinton", "Transit Days from Spring Grove", "Transit Days from Franklin Park"]

for column_name in df.columns:
    if column_name not in numerical_columns:
        categorical_columns.append(column_name)

df[categorical_columns] = df[categorical_columns].astype('category')
#df.info()

# %%
# Step not really needed anymore, there are no rows without a Company ID & Customer Name, we need at least one
# for index, row in df.iterrows():
    # if pd.isnull(row["Sold To Name"]) and pd.isnull(row["Customer ID"]):
        # print("Row index:", index)

# %%
# Find rows with missing customer IDs but non-missing company names
missing_customer_id_rows = df[df['Customer ID'].isnull() & df['Sold To Name'].notnull()]

for index, row in missing_customer_id_rows.iterrows():
    # Find rows with the same company name but a valid customer ID
    matching_rows = df[(df['Sold To Name'] == row['Sold To Name']) & df['Customer ID'].notnull()]
    # If matching rows are found, copy the customer ID from the first match
    if not matching_rows.empty:
        matching_customer_id = matching_rows.iloc[0]['Customer ID']
        df.at[index, 'Customer ID'] = matching_customer_id

# %%
empty_customer_id_count = df['Customer ID'].isnull().sum()
print("Number of empty customer IDs after filling in missing values:", empty_customer_id_count)

for index, row in df.iterrows():
    if pd.isnull(row['Customer ID']):
        print(row)

# %% [markdown]
# ### THERE ARE NO NULL CUSTOMER IDs & SOLD TO NAME
# Essentially this means that we have a known customer for every entry, and this required for looking at customer segments.

# %% [markdown]
# ## Starting Feature Engineering Missing Segments in the Data

# %%
df.fillna(df.median(), inplace=True) # For Numerical fill in with the Median Value
# df.fillna(df.mode().iloc[0], inplace=True) # For Categorical Fill in with the Mode
df.head(2)


# %%
def remove_outliers(df, column):
    mean = df[column].mean()
    std = df[column].std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers


# %%
removed_rows = pd.DataFrame(columns=df.columns)

for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        outliers = remove_outliers(df, column)
        removed_rows = removed_rows.append(outliers, ignore_index=True)
        df = df.drop(outliers.index)

# %%
print("Removed Rows:")
print(removed_rows)

# %% [markdown]
# # Preliminary EDA
#
# This is just me messing around with the data. I think it is somewhat helpful to read through and mess with these graphs yourself. Please excuse my "vagueness." I'm trying to obfuscate the data in my explanations. Just run the graphs and it'll make sense.

# %% [markdown]
# ## Grades
# The first grade makes up most of the entries. In the later 

# %%
plt.figure(figsize=(10, 10))
sns.countplot(gnw["Grade Family"]).set_title("Grade Family Freq")

# %% [markdown]
# There are a few grades_df that are only used by only a select few companies.

# %%
grade_enc = LabelEncoder()
grade_enc.fit(gnw["Grade Family"])

fig, ax = plt.subplots(3, layout='tight', figsize=(15, 15))
ax[0].scatter(grade_enc.transform(gnw["Grade Family"]), gnw["Customer ID"])
ax[0].set_title("Grade Family vs. Customer ID")
ax[0].set_xlabel("Customer ID")
ax[0].set_ylabel("Grade Family")
grade_enc.fit(gnw["Grade"])
ax[1].scatter(grade_enc.transform(gnw["Grade"]), gnw["Customer ID"])
ax[1].set_title("Grade vs Customer ID")
ax[1].set_xlabel("Customer ID")
ax[1].set_ylabel("Grade")
ax[2].scatter(gnw["Material Density"], gnw["Customer ID"])
ax[2].set_title("Material Density vs. Customer ID")
ax[2].set_xlabel("Customer ID")
ax[2].set_ylabel("Material Density")

# %% [markdown]
# ## Customers
#
# A logarithmic scale is used so that the less common country codes are visible. Play close attention to the scale.

# %%
plt.figure(figsize=(20,10))
plt.yscale('log')
sns.countplot(customers["Country Code"]).set_title("Most Common Country Codes")

# %% [markdown]
# # Conclusion
#
# There are a lot of null entries that likely need to be imputed unless someone has a better idea. The data is merged into a single dataframe, however this is not the only way to merge the data. A key limitation is that customer ID is hardly unique in the current dataframe. (Compare the unique customer ids to the length of the dataframe).
#
# However at least for EDA this is a powerful dataframe to use for the time being. When we start to engineer features it may be time to create a new dataframe structure.
