import pandas as pd
import numpy as np
import time
from itertools import combinations
import swifter
import ast
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.optim import AdamW
import networkx as nx
from networkx.algorithms.community import louvain_communities
from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
from rapidfuzz import fuzz
import re
from utils import upload_data
from pathlib import Path







# Function to remove the first {number}gb/ pattern
def remove_first_gb_pattern(name):
    return re.sub(r'\d+gb/', '', name, flags=re.IGNORECASE)



def prep(name, manufacturer, catID, category_tree_with_translation, stopwords):
    # Remove text within parentheses and strip whitespace
    name = re.sub(r'\(.*?\)', '', name).strip().lower()
    
    # Get all values of ct5.Name for the given catID
    ct5_names = category_tree_with_translation.loc[
        category_tree_with_translation['CategoryID'] == catID, 'ct5.Name'
    ].tolist()
    
    # Remove each ct5.Name value from the product name
    for ct5_name in ct5_names:
        if pd.notna(ct5_name):  # Check if the value is not NaN
            ct5_name = str(ct5_name).strip().lower()
            name = re.sub(r'\b' + re.escape(ct5_name) + r'\b', '', name)
    
    # Define replacements for cleaning the name
    replacements = {
        ',': ' ',
        '/': ' ',
        '.': ' ',
        '`': '',
        '()': '',
        '(2024)': ' ',
        '(2023)': ' ',
        '(2022)': ' ',
        '+': ' plus',
        'auxru': ' aux',
        'ps5': ' playstation 5',
        'ps4': ' playstation 4',
        'e-sim': ' ',
        'e-': ' ',
        'dual sim': ' ',
        'dual': ' ',
        'gb': ' ',
        '4g': ' ',
        '5g': ' ',
        'ё': 'е',
        'nfc': ' ',
        '   ': ' ',
        '  ': ' ',
        '\"': ' ',
        ',': ' ',
        '()': '',
        '+': 'plus',
        'auxru': ' aux',
        'ps5': ' playstation 5',
        'ps4': ' playstation 4',
        'ё': 'е',
        ' pro ': ' pro pro pro ',
        ' mini ': ' mini mini mini ',
        ' max ': ' max max max ',
        ' plus ': ' plus plus plus ',
        'hd08': 'hd08 hd08 hd08 hd08 hd08',
        'hd07': 'hd07 hd07 hd07 hd07 hd07',
        '-': ' '
    }

    # Special case: If CategoryID == 3 and Manufacturer == "apple", apply remove_first_gb_pattern
    if catID == 3 and pd.notna(manufacturer) and manufacturer.strip().lower() == 'apple':
        name = remove_first_gb_pattern(name)

    # Additional cleaning for categories 259 and 260
    if catID in [259, 260]:
        # Replace patterns like "number ml", "numberмл", etc., with "number_mlnumber_mlnumber_ml"
        name = re.sub(r'(\d+)([\s_-]?мл|[\s_-]?ml)', r'\1_ml\1_ml\1_ml', name, flags=re.IGNORECASE)

    # Remove manufacturer from the product name if it's not NaN
    if pd.notna(manufacturer):
        manufacturer = str(manufacturer).strip().lower()
        # Use a regex pattern to remove manufacturer regardless of position
        name = re.sub(r'\b' + re.escape(manufacturer) + r'\b', '', name)


    # Apply replacements to clean the name
    for x, y in replacements.items():
        name = name.replace(x, y)

    # Normalize name to handle stopword boundaries effectively
    name = re.sub(r'[^\w\s]', ' ', name)  # Replace non-word characters with spaces
    name = re.sub(r'\s+', ' ', name).strip()

    # Remove stopwords as separate words
    for word in stopwords:
        # Use a regex pattern to remove stopwords regardless of position
        name = re.sub(r'\b' + re.escape(word) + r'\b', '', name)
    
    
    # Clean up any extra spaces and return the cleaned name
    name = re.sub(r'\s+', ' ', name).strip()
    return name



# Define function to extract numeric values
def extract_numeric(value):
    if pd.isna(value):  # Handle NaN values
        return None
    return ''.join(filter(str.isdigit, str(value)))  # Keep only digits







def filter_data_by_category_and_characteristics(item: pd.Series, df: pd.DataFrame, selected_characteristics: list) -> list:
    """Filters the DataFrame based on category and selected characteristics."""
    data = df[df['CategoryID'] == item['CategoryID']]
    data = df[(df.CategoryID == item.CategoryID)]


    # Filter by Manufacturer if available
    if pd.notna(item['Manufacturer']):
        data = data[data['Manufacturer'].isna() | (data['Manufacturer'] == item['Manufacturer'])]

    # Filter by selected characteristics
    for col in selected_characteristics:
        if pd.notna(item[col]):
            data = data[data[col].isna() | (data[col] == item[col])]
    
    return [item['product_name_cln']] if data.empty else sorted(set(data['product_name_cln'].tolist() + [item['product_name_cln']]))

   

def process_and_generate_candidates(params, umico, chars, offers, parsed, default_threshold, category_tree_with_translation, stopwords):
    
    """Prepares data for the product matching task."""
    # Convert to lowercase, strip whitespace, and filter out non-string values
    stopwords = [w.strip().lower() for w in stopwords if isinstance(w, str)]
    # Sort the stopwords by word length in descending order
    stopwords = sorted(stopwords, key=lambda x: len(x), reverse=True)

    print(f'Stopwors: {stopwords}')

    # Initialize and process configurations
    selected_characteristics = list(sorted(set(params.get('chars_mapping').values())))
    category_dict = params.get('category', {})
    for_categories = pd.DataFrame([{"CategoryID": int(category_id), "threshold": threshold} 
                                   for category_id, threshold in category_dict.items()])
    for_categories['threshold'] = for_categories['threshold'].fillna(default_threshold)

    # Filter and merge Umico data
    umico = process_umico_data(umico, for_categories)
    
    # Process characteristics and merge with Umico data
    chars = process_characteristics(chars, umico, selected_characteristics, params)
    umico = umico.merge(chars, on='ProductID', how='left')

    # Process offers data
    offers = process_offers_data(offers)

    # Merge offers with Umico
    umico = umico.merge(offers, on='ProductID', how='left')

    # Process parsed data
    # parsed = process_parsed_data(parsed)

    # Concatenate Umico and Parsed DataFrames
    umico_n_parsed = concat_umico_parsed_data(umico, parsed, category_tree_with_translation, stopwords)

    # Matching preparation
    matching_df = prepare_matching_df(umico_n_parsed)
    
    # Perform filtering and timing
    matching_df = perform_filtering(matching_df, selected_characteristics)
    
    candidates = generate_product_pairs(matching_df)

    return umico_n_parsed, candidates


def process_umico_data(umico: pd.DataFrame, for_categories: pd.DataFrame) -> pd.DataFrame:
    """Processes the Umico DataFrame."""
    umico = umico[['CategoryID', 'ProductID', 'ProductName', 'Manufacturer', 'Status','AvailCheck']]
    umico = umico[umico['CategoryID'].isin(for_categories['CategoryID'])]
    umico['url'] = 'https://umico.az/product/'+umico['ProductID'].astype(str)
    umico['source'] = 'umico'



    return umico

def process_characteristics(chars: pd.DataFrame, umico: pd.DataFrame, selected_characteristics: list, params) -> pd.DataFrame:
    """Processes characteristics data and merges with Umico."""
    chars = chars[chars['product_id'].isin(umico['ProductID'])].sort_values(by='Created', ascending=False)
    chars = chars.dropna(subset=['merchant_field_name', 'values'])
    chars = chars[['product_id', 'merchant_field_name', 'values']].rename(columns={'product_id': 'ProductID'})
    chars['merchant_field_name'] = chars['merchant_field_name'].map(params.get('chars_mapping')).fillna(chars['merchant_field_name'])
    chars = chars[chars['merchant_field_name'].isin(selected_characteristics)]
    chars['values'] = chars['values'].str.replace(r'[\["\]]', '', regex=True)
    chars = chars.drop_duplicates(subset=['ProductID', 'merchant_field_name'], keep='first').reset_index(drop=True)
    chars = chars.pivot(index='ProductID', columns='merchant_field_name', values='values').reset_index()
    
    return chars


def process_offers_data(offers: pd.DataFrame) -> pd.DataFrame:
    """Filters and processes offers data."""
    offers['ProductID'] = offers['ProductID'].astype(int)
    active_offers = offers[(offers['Active'] == True) & (offers['Availability'] == '+')].sort_values('RetailPrice')
    active_offers = active_offers.drop_duplicates('ProductID', keep='first')
    not_active_offers = offers[~offers['ProductID'].isin(active_offers['ProductID'])].sort_values('RetailPrice')
    not_active_offers['RetailPrice'] = None
    not_active_offers = not_active_offers.drop_duplicates('ProductID', keep='first')
    return pd.concat([active_offers, not_active_offers])[['ProductID', 'Availability', 'RetailPrice']]





def concat_umico_parsed_data(umico: pd.DataFrame, parsed: pd.DataFrame,category_tree_with_translation, stopwords) -> pd.DataFrame:
    """Concatenates Umico and Parsed data."""

    # Define the required columns for both umico and parsed
    umico_columns = ['source', 'CategoryID', 'ProductName', 'url', 'RetailPrice', 'Manufacturer', 'ram', 'rom', 'color', 'series', 'volume', 'ProductID', 'Status', 'AvailCheck', 'Availability']
    parsed_columns = ['item_source', 'item_category', 'item_name', 'item_url', 'item_price', 'brand', 'ram', 'rom', 'colour']
    
    # Ensure missing columns in both DataFrames are replaced with empty columns
    for col in umico_columns:
        if col not in umico.columns:
            umico[col] = None  # Add an empty column if not found
    
    for col in parsed_columns:
        if col not in parsed.columns:
            parsed[col] = None  # Add an empty column if not found
    
    # Now select the relevant columns from both DataFramesumico_n_parsed = pd.concat([umico, parsed]).drop_duplicates().ap
    umico = umico[umico_columns]
    parsed = parsed[parsed_columns]
    parsed.columns = umico.columns[:len(parsed.columns)]
    # print(parsed)

    umico_n_parsed = pd.concat([umico, parsed]).drop_duplicates().apply(lambda col: col.astype(str).str.lower().str.strip() if col.dtypes == 'object' else col).reset_index(drop=True)
    
       
    # Apply the function only to rows where CategoryID == 3 and Manufacturer == 'Apple'
    # umico_n_parsed.loc[(umico_n_parsed['CategoryID'] == 3) & (umico_n_parsed['Manufacturer'].str.lower() == 'apple'), 'ProductName'] = umico_n_parsed['ProductName'].apply(remove_first_gb_pattern)
  
    # print(umico_n_parsed)
    # Apply the prep function while keeping the original ProductName
    umico_n_parsed['product_name_cln'] = umico_n_parsed.apply(lambda row: prep(row['ProductName'], row['Manufacturer'], row['CategoryID'], category_tree_with_translation, stopwords), axis=1)
    # print(umico_n_parsed)
    
    umico_n_parsed.loc[(umico_n_parsed['source'] == 'umico') & (umico_n_parsed['CategoryID'] == 3), 'color'] = np.nan

    return umico_n_parsed


def prepare_matching_df(umico_n_parsed: pd.DataFrame) -> pd.DataFrame:
    """Prepares the matching DataFrame."""
    umico_n_parsed['ram'] = umico_n_parsed['ram'].apply(extract_numeric)
    umico_n_parsed['rom'] = umico_n_parsed['rom'].apply(extract_numeric)
    umico_n_parsed['volume'] = umico_n_parsed['volume'].apply(extract_numeric)
    umico_n_parsed = umico_n_parsed[umico_n_parsed['CategoryID']!=16]
    umico_n_parsed = umico_n_parsed.reset_index(drop=True)
    return umico_n_parsed[['CategoryID', 'ProductName','product_name_cln',  'Manufacturer', 'ram', 'rom', 'color', 'series', 'volume']]

def perform_filtering(matching_df: pd.DataFrame, selected_characteristics: list) -> pd.DataFrame:
    """Applies filtering using swifter."""
    matching_df=matching_df.copy()
    matching_df.loc[:,'candidates_filter'] = matching_df.swifter.apply(lambda row: filter_data_by_category_and_characteristics(row, matching_df, selected_characteristics), axis=1)
    return matching_df
 
def generate_product_pairs(matching_df: pd.DataFrame) -> pd.DataFrame:
    """Generates product pairs from matching candidates."""
    candidates = matching_df['candidates_filter'].drop_duplicates().reset_index(drop=True)
    candidates = [sublist for sublist in candidates if len(sublist) > 1]  # Filter out lists with only one product name
    pairs = [pair for sublist in candidates for pair in combinations(sublist, 2)]  # Generate all combinations
    candidates_df = pd.DataFrame(pairs, columns=["product", "neighbor"])
    candidates_df["pair"] = candidates_df.apply(lambda row: tuple(sorted((row["product"], row["neighbor"]))), axis=1)
    return candidates_df.drop_duplicates(subset="pair").drop(columns="pair").reset_index(drop=True)










def load_model_and_predict(model_path, tokenizer_path, candidates_df, batch_size=32):
    class MatchingModel(nn.Module):
        def __init__(self, base_model):
            super(MatchingModel, self).__init__()
            self.base_model = base_model
            self.classifier = nn.Linear(base_model.config.hidden_size, 1)  # Binary classification

        def forward(self, input_ids, attention_mask):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            logits = self.classifier(pooled_output)
            return logits

    # Load the base model and tokenizer
    base_model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
    model = MatchingModel(base_model)
    # model.load_state_dict(torch.load(model_path))

    # Load the state dictionary from the file, mapping it to the CPU
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Prepare test pairs
    test_pairs = list(zip(candidates_df["product"], candidates_df["neighbor"]))
    predictions = []
    scores = []
    print(f'pairs: {len(test_pairs)}')
    # Process in batches
    for i in range(0, len(test_pairs), batch_size):
        print(f"Processing batch {i // batch_size + 1}/{(len(test_pairs) + batch_size - 1) // batch_size}")
    
        batch_pairs = test_pairs[i:i+batch_size]
        test_inputs = tokenizer(
            [f"{p[0]} [SEP] {p[1]}" for p in batch_pairs],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = model(input_ids=test_inputs["input_ids"], attention_mask=test_inputs["attention_mask"]).squeeze(-1)
        
        probabilities = torch.sigmoid(logits)
        batch_predictions = (probabilities > 0.5).int()

        predictions.extend(batch_predictions.cpu().numpy())
        scores.extend(probabilities.cpu().numpy())

    # Add predictions and scores to DataFrame
    candidates_df['predicted_match'] = predictions
    candidates_df['score'] = scores

    return candidates_df

















def extract_part_number(text):
    text = text.lower().strip().replace(' ','').replace('.','').replace('-n', '').replace('/','').replace('\\','').replace('-','')
    pattern = r'\(([A-Za-z0-9-]+(?:[a-zA-Z0-9-]*))\)'
    matches = re.findall(pattern, text)
    # Filter out anything that looks like a resolution (e.g., "1920x1080")
    filtered_matches = [match for match in matches if not re.match(r'\d{4}x\d{4}', match)]
    return filtered_matches[-1] if filtered_matches else text

def fill_missing_groups(df):
    """
    Fills missing group values in a DataFrame based on CategoryID.

    Args:
        df: The DataFrame to process. Must contain 'CategoryID', 'group', and 'ProductName' columns.

    Returns:
        The DataFrame with missing group values filled.
    """

    if df['CategoryID'].eq(16).any(): 
        df.loc[df['CategoryID'] == 16, 'group'] = df.loc[df['CategoryID'] == 16, 'group'].fillna(df.loc[df['CategoryID'] == 16, 'ProductName'].apply(extract_part_number))
    else:
        df['group'] = df['group'].fillna(df['ProductName'])

    return df



def process_product_matching(candidates, umico_n_parsed, threshold=0.95):
    # Create an empty graph
    G = nx.Graph()
    
    # Filter for matches only
    df_matches = candidates[candidates['predicted_match'] == 1]
    
    # Add edges for products with similarity above the threshold
    for _, row in df_matches.iterrows():
        if row['score'] > threshold:
            G.add_edge(row['product'], row['neighbor'], weight=row['score'])
    
    # Apply the Louvain method for community detection
    communities = list(louvain_communities(G))
    
    # Convert communities to DataFrame
    data = []
    for i, community in enumerate(communities, start=1):
        for product in community:
            data.append({'product_name_cln': product, 'group': i})
    
    groups = pd.DataFrame(data)
    
    # Merge the detected groups with the original dataset
    umico_n_parsed = umico_n_parsed.merge(groups, on='product_name_cln', how='left')

    umico_n_parsed = fill_missing_groups(umico_n_parsed)
    # If CategoryID is 16 and brand is 'apple'
    umico_n_parsed.loc[(umico_n_parsed['CategoryID'] == 16) & (umico_n_parsed['Manufacturer'] == 'apple'), 'group'] = umico_n_parsed.loc[(umico_n_parsed['CategoryID'] == 16) & (umico_n_parsed['Manufacturer'] == 'apple'), 'group'].str.slice(0, 5)


    # Fill missing values in 'group' column with cleaned product name
    umico_n_parsed['group'] = umico_n_parsed['group'].fillna(umico_n_parsed['product_name_cln'] + umico_n_parsed['CategoryID'].astype(str) + umico_n_parsed['Manufacturer'].fillna(''))
    umico_n_parsed['group'] = umico_n_parsed['group'].astype(str)
    # Encode the 'group' column
    le = LabelEncoder()
    umico_n_parsed['group'] = le.fit_transform(umico_n_parsed['group'])

    # Add current date as a proper date object
    today = datetime.today().date()  # Only date, no time
    umico_n_parsed['date'] = today  # Assign to DataFrame

    # Convert to string in SQL-compatible format (YYYY-MM-DD)
    umico_n_parsed['date'] = umico_n_parsed['date'].astype(str)
    umico_n_parsed = umico_n_parsed.sort_values(by=['date','group'])


    umico_n_parsed['RetailPrice'] = pd.to_numeric(umico_n_parsed['RetailPrice'], errors='coerce')
    umico_n_parsed['CategoryID'] = umico_n_parsed['CategoryID'].astype(int)
    umico_n_parsed['RetailPrice'] = umico_n_parsed['RetailPrice'].apply(
    lambda x: 999999999999999 if x > 999999999999999 else x
)
    umico_n_parsed['RetailPrice'] = pd.to_numeric(umico_n_parsed['RetailPrice'], errors='coerce')


    # Truncate string columns to max 499 characters for writing to DWH
    for col in umico_n_parsed.select_dtypes(include=['object', 'string']).columns:
        umico_n_parsed[col] = umico_n_parsed[col].astype(str).str.slice(0, 499)


    return umico_n_parsed






def rapidfuzz_cutoff(candidates, threshold):
    # candidates['product_cln'] = candidates['product'].apply(prep)
    # candidates['neighbor_cln'] = candidates['neighbor'].apply(prep)
    # Calculate fuzzy similarity scores
    print(f"\033[1m\033[93m{'=' * 50}")
    print(f'LENGTH OF CANDIDATES: {len(candidates)}')
    print(f"{'=' * 50}\033[0m")

    candidates['fuzzy_score'] = candidates.apply(lambda x: fuzz.ratio(x['product'], x['neighbor']), axis=1)
    candidates = candidates.sort_values(by='fuzzy_score')
    candidates.to_csv('fuzzy.csv', index=False)

    upload_data(
        datastore_name="mp_ds_matching",
        files=["fuzzy.csv" ],
        target_path="output/debug")
    
    candidates = candidates[candidates['fuzzy_score']>=threshold]
    candidates = candidates[['product','neighbor']].reset_index(drop=True)
    candidates = candidates.drop_duplicates().reset_index(drop=True)
    
    print(f"\033[1m\033[93m{'=' * 50}")
    print(f'LENGTH OF CANDIDATES: {len(candidates)}')
    print(f"{'=' * 50}\033[0m")

    return candidates



































def concat_last_n_days_data(historical, n_days):
    """
    Concatenates the last n days of historical data from CSV files in the specified directory.

    Args:
        historical: The directory containing the CSV files.
        n_days: The number of days of data to concatenate.

    Returns:
        A pandas DataFrame containing the concatenated data.
    """

    today = datetime.today().date()


    date_format = '%d.%m.%Y'

    file_paths = []
    for i in range(n_days):
        date = today - timedelta(days=i)
        file_name = f"matched_df_{date.strftime(date_format)}.csv"
        file_path = Path(historical) / file_name
        file_paths.append(file_path.resolve().as_posix())

    dfs = []
    missing_dates = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            missing_dates.append(date)

    if dfs:
        concatenated_df = pd.concat(dfs, ignore_index=True)
        if missing_dates:
            print(f"Missing data for the following dates: {', '.join(map(str, missing_dates))}")
        return concatenated_df
    else:
        print("No data found for the specified number of days.")
        return None
