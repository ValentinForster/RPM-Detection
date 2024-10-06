import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import entropy
import csv
import time
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu
from sklearn.metrics import silhouette_score

st.title("RPM detection tool")

st.write("""
This tool will find indications of [Resale Price Maintenance (RPM)](https://en.wikipedia.org/wiki/Resale_price_maintenance) in price data. 
To use it, upload your data as csv. The data should ideally have prices without shipping and should only only contain prices for one country/state.
Each row should represent one offering by one vendor for a specific product. Your csv should look like this:
""")

example_df = {"Manufacturer": ["Samsung", "Samsung", "Apple", "..."],
              "Product": ["Galaxy A25", "Galaxy A25", "IPhone 15 (256 GB)", "..."],
              "Vendor": ["Amazon", "Walmart", "Alibaba", "..."],
              "Price": [190.85, 195.00, 795.99, "..."],
              "Date": ["2024-08-26", "2024-08-26", "2024-08-24", "..."]}

st.dataframe(example_df, hide_index=True)

# Offer the CSV file for download as bytes
with open("Gefrierschränke/Gefrierschränke.csv", 'rb') as f:
    st.download_button(
        label="Download sample data",
        data=f,
        file_name='Freezer_prices_in_Austria_2024.csv',
        mime='text/csv',
    )

uploaded_file = st.file_uploader("Upload your CSV file to get it analyzed regarding RPM", type="csv")

if uploaded_file is not None:
    # Display a "Calculating..." message while the calculation is happening
    with st.spinner("Calculating... this may take a minute or two..."):
        # Data cleaning ------------------------------------------------------------------------------------------------------------------------------------------
        def detect_delimiter(uploaded_file):
            # Read the file buffer
            file_content = uploaded_file.getvalue().decode('utf-8')
            sniffer = csv.Sniffer()
            return sniffer.sniff(file_content[:5000]).delimiter
    
        # To adjust for different column names in the CSV files
        column_mapping = {
            'Model': ['Model', 'Product', 'Item'],
            'Vendor': ['Vendor', 'Seller', 'Retailer'],
            'Price_w/o_shipping': ['Price_w/o_shipping', 'Price', 'Cost'],
            'Date': ['Date', 'Timestamp', 'SaleDate']
        }
    
        delimiter = detect_delimiter(uploaded_file)
    
        df = pd.read_csv(uploaded_file, sep=delimiter, dtype=str)
    
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        for standard_name, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name.lower() in df.columns:
                    df.rename(columns={possible_name.lower(): standard_name}, inplace=True)
                    break
    
        # Convert Price column correct format. If the price is already in correct format, this will have no effect
        df["Price_w/o_shipping"] = (
            df["Price_w/o_shipping"]
            .str.replace(r'[^\d.,]', '', regex=True)  # Remove any non-numeric characters except commas and periods
            .str.replace(",", ".")                    # Replace comma with period
            .astype(float)
        )
        
        # Create column for manufacturer by taking first word of Model name (only if there is no Manufacturer column already in the dataset)
        if 'Manufacturer' not in df.columns:
            df["Manufacturer"] = df["Model"].str.split().str[0]
        
        # Drop all columns except the ones we need
        df = df[['Model','Price_w/o_shipping','Vendor', 'Manufacturer', 'Date']]
        
        # Drop rows with NA
        print(f"Initial number of rows: {len(df)}")
        df = df.dropna()
        print(f"After dropping NA: {len(df)}")
        
        # Delete rows where Manufacturer name is in Vendor (meaning that the vendor is a store operated by the manufacturer). No RPM possible when vendor and manufacturer is the same company
        df['Manufacturer'] = df['Manufacturer'].astype(str)
        df['Vendor'] = df['Vendor'].astype(str)
        df = df[~df.apply(lambda row: row['Manufacturer'] in row['Vendor'], axis=1)]
        print(f"After dropping offers by the manufacturer themselves: {len(df)}")
        
        # Convert all vendors to lowercase, to avoid one vendor being counted multiple times because of capitalization differences
        df['Vendor'] = df['Vendor'].apply(lambda x: x.lower())
        
        # Drop duplicates
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"After dropping duplicates: {len(df)}")
        
        # Exclude rows where the same vendor offers the same model at the same date for different prices (there may be variants of the model, like different colors). Keep the lowest price
        idx = df.groupby(['Model', 'Vendor', 'Date'])['Price_w/o_shipping'].idxmin()
        df = df.loc[idx].reset_index(drop=True)
        print(f"After dropping variants of each model: {len(df)}")
        
        # Calculate the number of vendors for each product
        df["Count_vendors"] = df.groupby(["Model", "Date"])["Model"].transform("count")
        
        # Add median, std, mean and coef_var per distinct model and date pair.
        df_grouped = df.groupby(['Model', 'Date']).agg(Median=('Price_w/o_shipping', 'median'), Std=('Price_w/o_shipping', 'std'), Mean=('Price_w/o_shipping', 'mean')).reset_index()
        df = df.merge(df_grouped, on=["Model","Date"])
        
        # Add Price_ratio (ratio of the price of the offering compared to the median & mean price)
        df['Price_ratio_median'] = df['Price_w/o_shipping'] / df['Median']
        df['Price_ratio_mean'] = df['Price_w/o_shipping'] / df['Mean']
        
        # Remove outliers (50% above median, 50% below median) --> if there are many timepoints in the dataset, this will remove a lot of data, as a typo by the vendor is removed multiple times
        df = df[df['Price_ratio_median'] <= 1.5]
        print(f"After removing outliers double the median price: {len(df)}")
        df = df[df['Price_ratio_median'] > 0.5]
        print(f"After removing outliers half the median price: {len(df)}")
        
        # Print counts of distinct vendors, models and manufacturers remaining in the dataset
        num_vendors = df['Vendor'].nunique()
        num_models = df['Model'].nunique()
        num_hersteller = df['Manufacturer'].nunique()
        print(f"Before Amthauer filtering, the dataset had {len(df)} rows, {num_vendors} vendors, {num_models} models from {num_hersteller} manufacturers")
        
        # ------------------------------------------------------------- Default settings of Amthauer et al. (2023) (excl. 3 Model rule) -----------------------------------------------------------------------------------
        
        # "We only include a model at one point in time if at this time it was offered by at least 5 vendors, to ensure that variation is generally possible." ------------------------------------------------------------
        df = df[df['Count_vendors'] >= 5]
        print(f"After 5 Vendor filtering, the dataset had {len(df)} rows")
        
        # Remove models, which are not offered at at least 75% of all distinct timepoints (only if not too much data is removed) ------------------------------------------------------------------------------------------
        
        # Calculate the total number of unique dates
        total_unique_dates = df['Date'].nunique()
        
        # Calculate 75% of the total number of unique dates
        threshold_75 = 0.75 * total_unique_dates
        
        # Group by model Model and count unique dates for each model
        model_date_counts = df.groupby('Model')['Date'].nunique()
        
        # Remove models that appear in less than 75% of all distinct dates
        below75_models = model_date_counts[model_date_counts < threshold_75].index.tolist()
        
        # If user uploads data spanning multiple product categories scraped on different dates, this would remove most data. Hence, this checks if much data would be removed. If yes, skip this step.
        if len(below75_models) * 0.25 < len(df):
            df = df[~df['Model'].isin(below75_models)]
            print(f"After 75% timepoints filtering, the dataset had {len(df)} rows ----------------------")
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Calculate the variance per model and timepoint pair across vendors. Calculate coef-var for each Model & Date combination
        vendor_var = df.groupby(['Model', 'Date'])['Price_w/o_shipping'].agg(['mean', 'std', 'nunique']).reset_index()
        vendor_var['coef_var_among_vendors'] = vendor_var['std'] / vendor_var['mean']
        vendor_var.rename(columns={"mean": "mean_among_vendors", 'std': 'std_among_vendors', 'nunique': 'distinct_vendor_prices'}, inplace=True)
        df = df.merge(vendor_var, on=["Model","Date"])
        
        # Drop rows with NA (just in case any NA were created during the previous steps)
        df = df.dropna()
        print(f"After dropping NA: {len(df)}")
        
        # Reorder columns
        new_column_order = ['Model',
                            'Price_w/o_shipping',
                            'Vendor',
                            'Count_vendors',
                            'Manufacturer',
                            'Date',
                            'Median',
                            'Mean',
                            'Std',
                            'Price_ratio_median',
                            'Price_ratio_mean',
                            'distinct_vendor_prices',
                            'mean_among_vendors',
                            'std_among_vendors',
                            'coef_var_among_vendors']
        
        df = df.reindex(columns=new_column_order)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------
        df['Date'] = pd.to_datetime(df['Date'])

        # Feature Engineering ----------------------------------------------------------------------------------------------------------------------------
        # Coef-Var among vendors represents the coef-var for each Model & Date combination. Calculate the average coef-var_among_vendors for each Model.
        avg_coef_var = df.groupby('Model')['coef_var_among_vendors'].mean().reset_index()
        avg_coef_var = avg_coef_var.rename(columns={'coef_var_among_vendors': 'mean_coef_var'})
    
        # Function to round prices based on 1% of the median price for each model
        def round_prices_to_one_percent(sub_df):
            model_median_price = sub_df['Median'].iloc[0]  # Get the median price for the model
            
            # Calculate 1% of the median price, and round it
            rounding_factor = model_median_price * 0.01
            
            if rounding_factor == 0:
                rounding_factor = 1  # Ensure that rounding factor is at least 1
            
            # Round the 'Price_w/o_shipping' to the nearest multiple of the rounding_factor
            sub_df['Price_w/o_shipping'] = (sub_df['Price_w/o_shipping'] / rounding_factor).round() * rounding_factor
            
            return sub_df
        
        # Apply the function to each group of models
        df = df.groupby('Model', group_keys=False).apply(round_prices_to_one_percent)
    
        # Check if the data contains dates spanning more than 30 days. If no, the num_price_changes column is not meaningful and we drop it in the end
        long_timeframe = (df['Date'].max() - df['Date'].min()).days >= 30
        
        # Group by 'Model', 'Vendor' and 'Manufacturer', then calculate the number of distinct prices. Thus, we know how many different prices the vendor charged for the model throughout the dataset.
        distinct_prices = df.groupby(['Model', 'Vendor', 'Manufacturer'])['Price_w/o_shipping'].nunique().reset_index()
        
        # Rename the column to 'Distinct_Price_Count'
        distinct_prices.columns = ['Model', 'Vendor', 'Manufacturer', 'Distinct_Price_Count']
        price_changes = distinct_prices.copy()
        price_changes['num_price_changes'] = distinct_prices['Distinct_Price_Count'] - 1
        
        # Calculate the average number of price changes for each model (across all vendors)
        price_changes = price_changes.groupby('Model')['num_price_changes'].mean().reset_index()
    
        # Group by Model, Date and count the occurences of value in Price_w/o_shipping
        price_counts = df.groupby(['Model', 'Date', 'Price_w/o_shipping']).size().reset_index(name='Count_of_prices')
        
        # Preparing df for next metric already
        price_counts['Min_Price'] = price_counts.groupby(['Model', 'Date'])['Price_w/o_shipping'].transform('min')
        price_counts['Is_Min_Price'] = price_counts['Price_w/o_shipping'] == price_counts['Min_Price']
        price_counts = price_counts[['Model', 'Date', 'Price_w/o_shipping', 'Count_of_prices', 'Is_Min_Price']]
        
        # Get the total number of offers per Model and Date
        offer_counts = df.groupby(['Model', 'Date']).size().reset_index(name='total_offers')
        
        # Calculate the maximum percentage of prices that are the same for each Model and Date
        same_price_pct = price_counts.groupby(['Model', 'Date'])['Count_of_prices'].apply(lambda x: x.max() / x.sum()).reset_index(name='same_price_pct')
        
        # Merge the percentage data with the total offers
        same_price_pct = pd.merge(same_price_pct, offer_counts, on=['Model', 'Date'])
        
        # Merge with manufacturer data
        manufacturer_model = df[['Model', 'Manufacturer']].drop_duplicates()
        same_price_pct = pd.merge(same_price_pct, manufacturer_model, on='Model')
        
        # Average the percentage of same prices for each model across all dates
        same_price_pct = same_price_pct.groupby(by='Model')['same_price_pct'].mean(numeric_only=False).reset_index()
    
        # Same as same_price_pct, but filter the price_counts df to only include the minimum prices
        cheap_same_price = price_counts[price_counts['Is_Min_Price'] == True]
        
        # Rename the column to 'count_cheap_same_price'
        cheap_same_price = cheap_same_price.rename(columns={'Count_of_prices': 'count_cheap_same_price'})
        
        # Merge with manufacturer data
        cheap_same_price = pd.merge(cheap_same_price, manufacturer_model, on='Model')
        
        # Merge with total offers
        cheap_same_price = pd.merge(cheap_same_price, offer_counts, on=['Model', 'Date'])
        
        # Calculate the percentage of the cheapest price for each Model and Date
        cheap_same_price['cheapest_same_price_pct'] = cheap_same_price['count_cheap_same_price'] / cheap_same_price['total_offers']
        
        # Average the percentage of the cheapest price for each model across all dates
        cheap_same_price = cheap_same_price.groupby(by='Model')['cheapest_same_price_pct'].mean(numeric_only=False).reset_index()
    
        # Function to calculate the entropy of prices for each model
        def calculate_entropy(prices):
            probabilities = np.bincount(prices) / len(prices) # Each unqiue value get own bin = maximum granularity (because we rounded before)
            return entropy(probabilities)
        
        # Group by 'Model' and calculate entropy for each group
        entropy_df = df.groupby('Model')['Price_w/o_shipping'].apply(lambda x: entropy(np.histogram(x, bins=len(x))[0], base=2)).reset_index()
        
        # Rename the columns
        entropy_df.columns = ['Model', 'entropy']
    
        results = pd.concat([avg_coef_var, price_changes, same_price_pct, cheap_same_price, entropy_df], axis=1)

        # Drop duplicate "Model" columns, keeping the first one
        results = results.loc[:,~results.columns.duplicated()]
        results = pd.merge(results, manufacturer_model, on='Model')
        
        # Drop the 'num_price_changes' column if the timeframe is not long enough
        if not long_timeframe:
            results.drop(columns=['num_price_changes'], inplace=True)

        # Clustering DBSCAN -------------------------------------------------------------------------------------------
        df = results

        # Exclude num_price_changes for clustering. Reasons: feature has high specificity but low sensitivity. Also, data spanning a short time span could make this feature unreliable
        features = df[["mean_coef_var", "same_price_pct", "cheapest_same_price_pct", "entropy"]]
        print(features.mean())
        
        # Standardizing the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(features)

        # Define the ranges for eps and min_samples to find the optimal parameters according to the Sillhouette Score
        eps_range = np.arange(0.1, 1.7, 0.1) # Zwischen 0.1 und 5 getestet: KS: 1, WS: 1.2, GefrierS: 1.1, GeschirrS: 0.7, Lautsprecher: 1.4. Aber: ohne Miele & Liebherr 0.1 optimal bei KS
        min_samples_range = range(3, len(df)//2) # min_samples cannot be greater than half of samples. Start with 3, since every manufacturer in the remaining data has at least 3 models
        
        best_eps = None
        best_min_samples = None
        best_score = -1
        
        # Find the best eps and min_samples
        for eps in eps_range:
            for min_samples in min_samples_range:
                db = DBSCAN(eps=eps, min_samples=min_samples)
                labels = db.fit_predict(scaled_data)
                
                # Calculate silhouette score if there is more than one cluster
                if len(set(labels)) > 1:
                    score = silhouette_score(scaled_data, labels)
                    if score > best_score:
                        best_eps = eps
                        best_min_samples = min_samples
                        best_score = score
        
        print(f"Best eps: {best_eps}")
        print(f"Best min_samples: {best_min_samples}")
        print(f"Best Silhouette Score: {best_score}")
        
        # Fit DBSCAN with best eps and min_samples
        db = DBSCAN(eps=best_eps, min_samples=best_min_samples)
        labels_dbscan = db.fit_predict(scaled_data)
        
        # Add cluster labels to the dataframe
        df['cluster_dbscan'] = labels_dbscan
        
        # Calculate Silhouette Score and print it
        dbscan_silhouette = silhouette_score(scaled_data, df['cluster_dbscan'])
        print(f"Silhouette Score for DBSCAN: {dbscan_silhouette}")

        # Get number of detected clusters
        cluster_labels = df['cluster_dbscan'].unique()
        print(f"Number of clusters: {len(cluster_labels)}")
        
        # If only one cluster is detected or more than 4 clusters are detected, there most likely is no RPM
        if len(cluster_labels) == 1 or len(cluster_labels) > 4:
            print("No RPM has been detected")
        else:
            # Create dict with cluster as key and 25th percentile coef_var as value. Reason: If the outlier cluster has at least 25% low outliers, we correctly identify it
            cluster_coefvar = {}
            for cluster in cluster_labels:
                cluster_coefvar[cluster] = df[df['cluster_dbscan'] == cluster]['mean_coef_var'].quantile(0.25)
            
            print(cluster_coefvar)
            print("-------------------------------------------------------------")
            
            # Get the clusters with the lowest and second lowest coef_var
            sorted_clusters = sorted(cluster_coefvar.items(), key=lambda x: x[1])
            lowest = sorted_clusters[0][0]
            second_lowest = sorted_clusters[1][0]
        
            # Make lowest cluster the suspicious cluster and second lowest the normal cluster
            outliers = df[df['cluster_dbscan'] == lowest]
            non_sus = df[df['cluster_dbscan'] == second_lowest]
        
            # Define fixed threshold as safety measure, since otherwise low outliers, which are only suspicious compared to the other products, are flagged
            non_sus_same_price_cutoff = 0.65
        
            # Just in case: Only keep low outliers in, remove high outliers from suspicious cluster, if there are any (normally not the case)
            print(f"These were removed from the suspicious cluster via the Same price cutoff: {non_sus_same_price_cutoff}")
            sus_only = outliers[outliers['same_price_pct'] >= non_sus_same_price_cutoff]
        
            alpha_level = 0.05
        
            # Initialize confidence variable, which gets higher if more tests are passed
            confidence = 0
        
            # Coef_var Mann-Whitney-U test: Mann-Whitney-U test because we have non-normal data and unequal variances
            print(f"Average coef var in sus cluster: {sus_only['mean_coef_var'].mean()}")
            print(f"Average coef var in normal cluster: {non_sus['mean_coef_var'].mean()}")
        
            sus_cluster_coefvar = sus_only['mean_coef_var']
            normal_cluster_coefvar = non_sus['mean_coef_var']
        
            t_coefvar, p_coefvar = mannwhitneyu(sus_cluster_coefvar, normal_cluster_coefvar, alternative='two-sided')
        
            print(f"P-value for coef-var: {round(p_coefvar, 2)} ({p_coefvar})")
        
            if p_coefvar < alpha_level:
                rpm_detected = True
                print("Result: One cluster has significantly lower variation in prices than the other(s)")
            else:
                print("Result: Both clusters have similar variation in prices")
        
            print("-------------------------------------------------------------")
            
            # Num_price_changes Mann-Whitney-U test ---------------------------------------------
        
            if 'num_price_changes' in df.columns:
                print(f"Average num_price_changes in sus cluster: {sus_only['num_price_changes'].mean()}")
                print(f"Average num_price_changes in normal cluster: {non_sus['num_price_changes'].mean()}")
        
                if sus_only['num_price_changes'].mean() >= non_sus['num_price_changes'].mean():
                    print("Result: The suspicious cluster has actually a higher number of price changes than the normal cluster")
                else:
                    sus_cluster_price_changes = sus_only['num_price_changes']
                    normal_cluster_price_changes = non_sus['num_price_changes']
        
                    t_changes, p_changes = mannwhitneyu(sus_cluster_price_changes, normal_cluster_price_changes, alternative='two-sided')
        
                    print(f"P-value for price changes: {round(p_changes, 2)} ({p_changes})")
        
                    if p_changes < alpha_level:
                        confidence += 1
                        print("Result: The same cluster also has significantly less price_changes than the other(s). Cluster likely represents RPM")
                    else:
                        print("Result: The same cluster DOES NOT have significantly less price_changes than the other")
            else:
                print("Time span too short to validate RPM cluster using num_price_changes")

        if rpm_detected:
            # Get the number of models per manufacturer
            value_counts = df['Manufacturer'].value_counts().reset_index()
            value_counts.columns = ['Manufacturer', 'total_models']
        
            # Group by manufacturer and count occurrences
            sus = sus_only.groupby('Manufacturer').size().reset_index(name='sus_count')
        
            # Merge the DataFrames
            result = pd.merge(value_counts, sus, on='Manufacturer', how='left').fillna(0)
            result['sus_count'] = result['sus_count'].astype(int)
            result['pct_suspicious_models'] = result['sus_count'] / result['total_models']
            result = result.sort_values(by=['pct_suspicious_models', 'sus_count'], ascending=False)
        
            # Only suspicious manufacturers (at least 5 sus models and more than 50 % sus models)
            sus_manufacturers = result[(result['sus_count'] > 4) & (result['pct_suspicious_models'] > 0.4)]
        
        sus_cluster_label = sus_only['cluster_dbscan'].unique()[0]

    st.title("Results")

    if len(sus_manufacturers) == 0:
                st.write("No suspicious manufacturers found.")

    st.write(f"There are {len(sus_only)} suspicious models, which is {len(sus_only) / len(df) * 100:.2f}% of the total models.")
    if confidence > 0:
        st.write("High confidence that the marked models are indicating RPM.")
    else:
        st.write("Medium confidence that the marked models are indicating RPM.")

    # Create a toggle for the user to switch between views
    view_option = st.radio("", ("Suspicious Manufacturers only", "All results"))
    
    # Define the filtered version of the DataFrame
    if view_option == "Suspicious Manufacturers only":
        st.dataframe(sus_manufacturers.head(2), hide_index=True)
        st.markdown("""**How to interpret the results**""")
        st.write("This table shows only those manufacturers in the dataset, which are likely using RPM. They have 5 or more suspicious products and more than 40 % of their products are suspicious.")
    else:
        # Display the unfiltered DataFrame
        st.dataframe(result, hide_index=True)
        st.markdown("""**How to interpret the results**""")
        st.write("This table shows the number as well as the percentage of suspicious models for each Manufacturer in the dataset. Please note that a high pct_suspicious_models with a small sample size (e.g. < 5 suspicious models) is not meaningful.")
  
    #st.dataframe(sus, hide_index=True)
