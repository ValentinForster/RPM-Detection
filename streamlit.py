import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import entropy
import csv
import time

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
        time.sleep(5) #--------------
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
    
        # Make Price column to type float. If the price is already in correct format, this will have no effect
        df["Price_w/o_shipping"] = (
            df["Price_w/o_shipping"]
            .str.replace(r'[^\d.,]', '', regex=True)  # Remove any non-numeric characters except commas and periods
            .str.replace(",", ".")                    # Replace comma with period
            .astype(float)
        )
    
        # Create column for manufacturer by taking first word of Model name (if it is not already in the dataset)
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
    
        # Exclude rows where the same vendor offers the same model at the same date for different prices (there may be variants of the model). Keep the lowest price
        idx = df.groupby(['Model', 'Vendor', 'Date'])['Price_w/o_shipping'].idxmin()
        df = df.loc[idx].reset_index(drop=True)
        print(f"After dropping variants of each model: {len(df)}")
    
        # Calculate the number of vendors for each product
        df["Count_vendors"] = df.groupby(["Model", "Date"])["Model"].transform("count")
    
        # Add median, std, mean and coef_var per distinct model and date pair.
        df_grouped = df.groupby(['Model', 'Date']).agg(Median=('Price_w/o_shipping', 'median'), Std=('Price_w/o_shipping', 'std'), Mean=('Price_w/o_shipping', 'mean')).reset_index()
        #df_grouped['coef_var'] = df_grouped['Std'] / df_grouped['Mean']
        df = df.merge(df_grouped, on=["Model","Date"])
    
        # Add Price_ratio (ratio of the price of the offering compared to the median & mean price)
        df['Price_ratio_median'] = df['Price_w/o_shipping'] / df['Median']
        df['Price_ratio_mean'] = df['Price_w/o_shipping'] / df['Mean']
    
        # Remove outliers (50% above median, 50% below median) --> about 700 rows of 1 Million removed (every typo counts several times, so that's to be expected)
        # Removes 11 rows for high outliers, none for low outliers
        df = df[df['Price_ratio_median'] <= 1.5]
        print(f"After removing outliers double the median price: {len(df)}")
        df = df[df['Price_ratio_median'] > 0.5]
        print(f"After removing outliers half the median price: {len(df)}")
    
        # Get counts of distinct vendors, models and manufacturers
        num_vendors = df['Vendor'].nunique()
        num_models = df['Model'].nunique()
        num_hersteller = df['Manufacturer'].nunique()
        print(f"Before Amthauer filtering, the dataset had {len(df)} rows, {num_vendors} vendors, {num_models} models from {num_hersteller} manufacturers")
    
        # ------------------------------------------------------------- Default settings of Amthauer et al. -------------------------------------------------------------
    
        # -------------"We only include a model at one point in time if at this time it was offered by at least 5 vendors, to ensure that variation is generally possible."
        df = df[df['Count_vendors'] >= 5]
        print(f"After 5 Vendor filtering, the dataset had {len(df)} rows")
    
        # -------------Remove models, which are not offered at at least 75% of all distinct timepoints
        # "After removing machines that are not offered on at least 75 % of the timepoints..."
    
        # Calculate the total number of unique dates
        total_unique_dates = df['Date'].nunique()
    
        # Calculate 75% of the total number of unique dates
        threshold_75 = 0.75 * total_unique_dates
    
        # Group by model Model and count unique dates for each model
        model_date_counts = df.groupby('Model')['Date'].nunique()
    
        # Remove models that appear in less than 75% of all distinct dates
        below75_models = model_date_counts[model_date_counts < threshold_75].index.tolist()
        df = df[~df['Model'].isin(below75_models)]
        print(f"After 75% timepoints filtering, the dataset had {len(df)} rows")
    
        # -------------"We then exclude manufacturers that have fewer than 3 models that are offered at any given point in time." --> 3 different models in total in the dataset, not min. 3 per timepoint!
        # Get those manufacturers, which offer less than 3 models
        model_count_per_hersteller = df.groupby('Manufacturer')['Model'].nunique().reset_index()
        three_less_models = model_count_per_hersteller[model_count_per_hersteller['Model'] < 3]
    
        # Leave only those manufacturers in, which don't appear in the three_less_models df
        df = df[~df['Manufacturer'].isin(three_less_models['Manufacturer'])]
        print(f"After 3 Models filtering, the dataset had {len(df)} rows")
    
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
        # Calculate the variance per model and timepoint pair across vendors. Essentially, how much does a model's price vary among vendors for each timepoint
        vendor_var = df.groupby(['Model', 'Date'])['Price_w/o_shipping'].agg(['mean', 'std', 'nunique']).reset_index()
        vendor_var['coef_var_among_vendors'] = vendor_var['std'] / vendor_var['mean']
        vendor_var.rename(columns={"mean": "mean_among_vendors", 'std': 'std_among_vendors', 'nunique': 'distinct_vendor_prices'}, inplace=True)
        df = df.merge(vendor_var, on=["Model","Date"])
    
        # There will be NA values in std_time_only. If the model is offered by >= 5 vendors at only one timepoint, all the other times the vendors offered the product will get lost. Hence, the std will be NA.
        # We don't want to fill those rows with 0, as the variance across time is likely non-zero, we just got rid of the remaining data
        df = df.dropna()
    
        # Create numerical codes for manufacturer, vendor, and model
        df['Manufacturer_code'] = pd.factorize(df['Manufacturer'])[0] + 1
        df['Vendor_code'] = pd.factorize(df['Vendor'])[0] + 1
        df['Model_code'] = pd.factorize(df['Model'])[0] + 1
    
        # Reorder columns
        new_column_order = ['Model',
                            'Model_code',
                            'Price_w/o_shipping',
                            'Vendor',
                            'Count_vendors',
                            'Manufacturer',
                            'Date',
                            'Vendor_code',
                            'Manufacturer_code',
                            'Median',
                            'Mean',
                            'Std',
                            'Price_ratio_median',
                            'Price_ratio_mean',
                            'mean_among_vendors',
                            'std_among_vendors',
                            'distinct_vendor_prices',
                            'coef_var_among_vendors']
    
        df = df.reindex(columns=new_column_order)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------
        df['Date'] = pd.to_datetime(df['Date'])
    
        avg_coef_var = df.groupby('Model')['coef_var_among_vendors'].mean().reset_index()
        avg_coef_var = avg_coef_var.rename(columns={'coef_var_among_vendors': 'mean_coef_var'})
    
        median_price = df['Price_w/o_shipping'].median()
    
        # Round prices so for example 799.90 € and 799.99 € are counted as the same price. Adjust rounding according to magnitude of prices, to adjust for high-price products or other currencies (e.g. japanese yen).
        if median_price < 500:
            df['Price_w/o_shipping'] = df['Price_w/o_shipping'].round()
        elif median_price < 1000:
            df['Price_w/o_shipping'] = (df['Price_w/o_shipping'] / 5).round() * 5
        elif median_price < 5000:
            df['Price_w/o_shipping'] = df['Price_w/o_shipping'].round(-1)
        elif median_price < 10000:
            df['Price_w/o_shipping'] = (df['Price_w/o_shipping'] / 50).round() * 50
        else:
            df['Price_w/o_shipping'] = df['Price_w/o_shipping'].round(-2)
    
        # Check if the data contains dates spanning more than 30 days. If no, the column is not meaningful and we drop it in the end
        long_timeframe = (df['Date'].max() - df['Date'].min()).days >= 30
    
        # Group by 'Model' and 'Vendor', then calculate the number of distinct prices
        distinct_prices = df.groupby(['Model', 'Vendor', 'Manufacturer'])['Price_w/o_shipping'].nunique().reset_index()
    
        # Rename the column for clarity
        distinct_prices.columns = ['Model', 'Vendor', 'Manufacturer', 'Distinct_Price_Count']
        price_changes = distinct_prices.copy()
        price_changes['num_price_changes'] = distinct_prices['Distinct_Price_Count'] - 1
    
        price_changes = price_changes.groupby('Model')['num_price_changes'].mean().reset_index()
    
        # Group by Model, Date and count the frequency of each Price_w/o_shipping
        price_counts = df.groupby(['Model', 'Date', 'Price_w/o_shipping']).size().reset_index(name='Count_of_prices')
        price_counts['Min_Price'] = price_counts.groupby(['Model', 'Date'])['Price_w/o_shipping'].transform('min')
        price_counts['Is_Min_Price'] = price_counts['Price_w/o_shipping'] == price_counts['Min_Price']
        price_counts = price_counts[['Model', 'Date', 'Price_w/o_shipping', 'Count_of_prices', 'Is_Min_Price']]
    
        # Get the total number of offers per Model and Date
        offer_counts = df.groupby(['Model', 'Date']).size().reset_index(name='total_offers')
    
        # Calculate the percentage of prices that are the same for each Model and Date
        same_price_pct = price_counts.groupby(['Model', 'Date'])['Count_of_prices'].apply(lambda x: x.max() / x.sum()).reset_index(name='same_price_pct')
    
        # Merge the percentage data with the total offers
        same_price_pct = pd.merge(same_price_pct, offer_counts, on=['Model', 'Date'])
    
        # Merge with manufacturer data
        manufacturer_model = df[['Model', 'Manufacturer']].drop_duplicates()
        same_price_pct = pd.merge(same_price_pct, manufacturer_model, on='Model')
    
        same_price_pct = same_price_pct.groupby(by='Model')['same_price_pct'].mean(numeric_only=False).reset_index()
    
        # Filter the price_counts df to only include the minimum prices
        cheap_same_price = price_counts[price_counts['Is_Min_Price'] == True]
        cheap_same_price = cheap_same_price.rename(columns={'Count_of_prices': 'count_cheap_same_price'})
        cheap_same_price = pd.merge(cheap_same_price, manufacturer_model, on='Model')
        cheap_same_price = pd.merge(cheap_same_price, offer_counts, on=['Model', 'Date'])
        cheap_same_price['cheapest_same_price_pct'] = cheap_same_price['count_cheap_same_price'] / cheap_same_price['total_offers']
        cheap_same_price = cheap_same_price.groupby(by='Model')['cheapest_same_price_pct'].mean(numeric_only=False).reset_index()
    
        # Function to calculate the entropy of prices for each model ------------------ Binning assumes integers????
        def calculate_entropy(prices):
            probabilities = np.bincount(prices) / len(prices)
            return entropy(probabilities)
    
        # Group by 'Model' and calculate entropy for each group
        entropy_df = df.groupby('Model')['Price_w/o_shipping'].apply(lambda x: entropy(np.histogram(x, bins=len(x))[0], base=2)).reset_index()
    
        # Rename the columns
        entropy_df.columns = ['Model', 'entropy']
    
        results = pd.concat([avg_coef_var, price_changes, same_price_pct, cheap_same_price, entropy_df], axis=1)
    
        # Drop duplicate "Model" columns, keeping the first one
        results = results.loc[:,~results.columns.duplicated()]
        results = pd.merge(results, manufacturer_model, on='Model')
    
        if not long_timeframe:
            results.drop(columns=['num_price_changes'], inplace=True)
    
        # Criteria to mark suspicious models
        mean_coefvar = results['mean_coef_var'].mean()
        changes_mean = results['num_price_changes'].mean()
    
        quantiles = results['mean_coef_var'].quantile([0.1, 0.25, 0.5, 0.75])
        iqr = quantiles[0.75] - quantiles[0.25]
    
        if mean_coefvar - 0.75 * iqr < mean_coefvar / 4:
            conditions = {
                'mean_coef_var': lambda x: x < (mean_coefvar / 4),
                'same_price_pct': lambda x: x >= 0.75,
                'cheapest_same_price_pct': lambda x: x >= 0.65,
                'entropy': lambda x: x <= 0.75
            }
        else:
            conditions = {
                'mean_coef_var': lambda x: x < (mean_coefvar - 0.75 * iqr),
                'same_price_pct': lambda x: x >= 0.75,
                'cheapest_same_price_pct': lambda x: x >= 0.65,
                'entropy': lambda x: x <= 0.75
            }
    
        # Apply the conditions and create new boolean columns
        for col, condition in conditions.items():
            results[col + '_sus'] = results[col].apply(condition)
    
        # Count the number of True values per row
        bool_columns = results.select_dtypes(include='bool')
        results['sus_count'] = bool_columns.sum(axis=1)
    
        new_column_order = ['Model', 'sus_count', 'Manufacturer', 'mean_coef_var', 'mean_coef_var_sus', 'same_price_pct', 'same_price_pct_sus', 'cheapest_same_price_pct', 'cheapest_same_price_pct_sus', 'entropy', 'entropy_sus']
    
        results = results.reindex(columns=new_column_order).sort_values(by='sus_count', ascending=False)
    
        # Filter the DataFrame for rows where sus_count is greater than 2
        df_filtered = results[results['sus_count'] > 2]
    
        # Group by manufacturer and count occurrences
        sus = df_filtered.groupby('Manufacturer').size().reset_index(name='sus_count')
    
        # Merge the DataFrames
        total_models = results.groupby('Manufacturer').size().reset_index(name='total_models')
        sus = pd.merge(sus, total_models, on='Manufacturer', how='left').fillna(0)
        sus['pct_suspicious_models'] = sus['sus_count'] / sus['total_models']
        sus = sus.sort_values(by='pct_suspicious_models', ascending=False)

    st.title("Results")

    # Create a toggle for the user to switch between views
    view_option = st.radio("", ("Suspicious Manufacturers only", "All results"))
    
    # Define the filtered version of the DataFrame
    if view_option == "Suspicious Manufacturers only":
        st.dataframe(sus.head(2), hide_index=True)
        st.markdown("""**How to interpret results**""")
        st.write("This table shows only those manufacturers from the dataset, which are likely using RPM. They have 5 or more suspicious products and over 40 % of their products are suspicious.")
    else:
        # Display the unfiltered DataFrame
        st.dataframe(sus, hide_index=True)
        st.markdown("""**How to interpret results**""")
        st.write("This table shows the number as well as the percentage of suspicious models for each Manufacturer in the dataset. Please note that a high pct_suspicious_models with a small sample size (e.g. < 5 suspicious models) is meaningless.")
  
    #st.dataframe(sus, hide_index=True)
