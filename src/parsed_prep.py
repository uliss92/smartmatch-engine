import pandas as pd
import os
import csv
import json
from openai import AzureOpenAI
from fuzzywuzzy import process
import re
import unicodedata
import ast
import numpy as np




def get_client(endpoint: str, key: str, api_version):
    os.environ["OPENAI_API_TYPE"] = "azure"
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=key)
    return client

# Function to extract category based on brand and the rest of the product name

def extract_category_from_name(product_name, brands):
    tokens = product_name.lower().split()
    for i, token in enumerate(tokens):
        if token in brands:
            # Assume the part before the brand token is the category
            assumed_category = " ".join(tokens[:i]).strip()
            if not assumed_category:
                assumed_category = None  # Avoid empty strings, use None instead
            print(f"Assumed category '{assumed_category}' for product '{product_name}' based on brand '{token}'")
            return assumed_category
    return None  # Return None if no brand is found




# Function to check if the value is numeric after removing spaces
def is_numeric_or_spaces(item):
    item_no_spaces = item.replace(' ', '').strip()  # Remove spaces and strip leading/trailing whitespace
    return item_no_spaces.isdigit() or item_no_spaces == ''  # Check if it's numeric or empty

def find_best_category(assumed_category, category_list, threshold=90):

    # Check for empty or None values before proceeding
    if not assumed_category or assumed_category.strip() == "":
        return None
    # Extract the best match with a threshold
    result = process.extractOne(assumed_category, category_list)

    if result:
        match, score = result[:2]  # Unpack only the match and score
        if score >= threshold:
            return match
    return None  # Return None if no match with the required threshold



# Function to apply the replacement map logic
def apply_replacement_map(assumed_cat, best_category, replacement_map):
    # Check if the assumed_cat is in the replacement_map
    if assumed_cat in replacement_map:
        # If it exists, update the best_category to the mapped value
        return replacement_map[assumed_cat]
    # Otherwise, return the original best_category
    return best_category

# Function to match a product to the best category and write to CSV
def find_best_category_and_write_to_csv(product_name, categories, csv_file):
    api_version = os.environ['OPENAI_API_VERSION']
    key = os.environ['OPENAI_API_KEY']
    endpoint = os.environ['OPENAI_API_ENDPOINT']
    deployment_name = os.environ['OPENAI_DEPLOYMENT_NAME']

    client = get_client(endpoint, key, api_version)

    # Prepare the prompt for the model
    prompt = f"You need to find the best suiting category for a product. It may be in Azerbaijani or Russian. If you are certain about the best category match, return it. If there is any doubt, return an empty JSON object. Given the product name '{product_name}', which categories from the following list best describes this product?\n\nCategories:\n" + ", ".join(categories)

    base_instruct = """
    Return the result in the following JSON format:

    {
        "Category": ["<CategoryID1>", "<CategoryID2>", ...]
    }
    
    Only return this JSON object. Do not include any additional text.
    """

    try:
        # Call GPT to get the response
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "user", "content": prompt + base_instruct}
            ],
            max_tokens=50
        )

        raw_response_content = response.choices[0].message.content.strip()

        # Clean up the response if it's not in pure JSON format
        if raw_response_content.startswith('```json'):
            raw_response_content = raw_response_content.replace('```json', '').replace('```', '').strip()

        # Check for empty JSON response
        if raw_response_content == '{}':
            print(f'No category match found for {product_name}. Returning empty list.')
            cat = []
        else:
            # Handle JSON response
            try:
                result_json = json.loads(raw_response_content)

                # Check if 'Category' key exists in the result
                if 'Category' in result_json and result_json['Category']:
                    cat = result_json['Category']
                    print(f'{product_name} to category {cat}')
                else:
                    cat = []  # No category found
                    print(f'No category match found for {product_name}')
            except json.JSONDecodeError:
                print(f"Invalid JSON response for {product_name}: {raw_response_content}")
                cat = []

        # Check if the CSV file exists to determine if we need to write headers
        file_exists = os.path.isfile(csv_file)

        # Write to CSV
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Product Name', 'Categories'])  # Write header if file is new
            writer.writerow([product_name, ", ".join(cat)])

        return cat  # Return the extracted category

    except Exception as e:
        print(f"An error occurred: {e}")
        with open('errors.log', 'a') as error_log:
            error_log.write(f"Error for {product_name}: {str(e)}\n")
        return []
    

def map_categories_to_parsed_products(mp_products_mpn_gtin, parsed,params, category_tree_with_translation, csv_file, replacement_map, n_cash):
    for_cats = list(params.get('category').keys())
    phones = parsed[parsed['item_category']==3]
    parsed = parsed[parsed['item_category']!=3]
    brands = list(mp_products_mpn_gtin['Manufacturer'].str.lower().unique())
    category_tree_with_translation['CategoryID'] = category_tree_with_translation['CategoryID'].astype(str)
    category_tree_with_translation = category_tree_with_translation[category_tree_with_translation['CategoryID'].isin(for_cats)]
    category_list_GPT = list(category_tree_with_translation[category_tree_with_translation['locale']=='ru']['ct5.Name'].astype(str)+': '+category_tree_with_translation[category_tree_with_translation['locale']=='ru']['CategoryID'].astype(str))
    category_tree_with_translation['ct5.Name'] = category_tree_with_translation['ct5.Name'].str.lower()
    category_list = category_tree_with_translation['ct5.Name']

    # Apply the filter to remove rows where 'item_name' is numeric or contains only spaces
    parsed['item_name'] = parsed['item_name'].str.replace('<[^>]*>', '', regex=True)
    parsed = parsed[~parsed['item_name'].apply(is_numeric_or_spaces)]
    parsed['item_name'] = parsed['item_name'].str.lower().str.strip()
    parsed['assumed_cat'] = parsed['item_name'].apply(lambda x: extract_category_from_name(x, brands))
    parsed['assumed_cat'] = parsed['assumed_cat'].str.strip()
    parsed['assumed_cat'] = parsed['assumed_cat'].str.replace('i̇', 'i', regex=False)
    parsed['best_category'] = parsed['assumed_cat'].apply(lambda x: find_best_category(x, category_list))
    parsed['best_category'] = parsed.apply(lambda row: apply_replacement_map(row['assumed_cat'], row['best_category'], replacement_map), axis=1)
    parsed = parsed.merge(category_tree_with_translation[['CategoryID','ct5.Name']], how='left', left_on='best_category', right_on='ct5.Name')
    parsed['item_category'] = pd.to_numeric(parsed['CategoryID'], errors='coerce')
    parsed=parsed.drop(columns=['assumed_cat','best_category','CategoryID','ct5.Name'])
    parsed['item_category'] = parsed['item_category'].astype('Int64')


    # Read past products
    past_products = pd.read_csv(csv_file)
    past_products = past_products[n_cash:]

    #past_products = past_products.dropna()
    past_products = past_products['Product']



    products = list(parsed[parsed['item_category'].isna()]['item_name'].unique())
    products = list(set(products) - set(past_products))
    print(f'Products to identify: {len(products)}')

    # Convert the list to a DataFrame
    df_products = pd.DataFrame(products, columns=['Product'])

    # Apply the function to each product, filtering categories by product name tokens
    df_products['item_category'] = df_products['Product'].apply(
        lambda product: find_best_category_and_write_to_csv(
            product, 
            category_list_GPT,
            csv_file
        )
    )


    # Remove duplicates and save the CSV
    df_product_category_mapping = pd.read_csv(csv_file)
    df_product_category_mapping = df_product_category_mapping[n_cash:]
    df_product_category_mapping.drop_duplicates(subset=['Product']).to_csv(csv_file, index=False)
    # Reload the CSV to ensure changes are saved
    df_product_category_mapping = pd.read_csv(csv_file)
    # Convert the CategoryID column to strings and handle NaNs or other invalid types
    df_product_category_mapping['Category'] = df_product_category_mapping['Category'].astype(str)
    df_product_category_mapping['Category'] = df_product_category_mapping['Category'].str.split(', ')
    df_product_category_mapping = df_product_category_mapping.explode('Category').reset_index(drop=True)
    df_product_category_mapping['Category'] = pd.to_numeric(df_product_category_mapping['Category'], errors='coerce')
    # df_product_category_mapping['Category'] = df_product_category_mapping['Category'].astype('int64')

    processed = parsed[parsed['item_category'].isna()].merge(df_product_category_mapping, how='left', left_on='item_name', right_on='Product')
    processed['item_category'] = processed['Category']
    processed = processed.drop(columns=['Product', 'Category'])
    parsed = pd.concat([parsed[parsed['item_category'].notna()],processed])
    # Specify all columns except 'item_specs'
    subset_columns = parsed.columns.difference(['item_specs'])

    # Drop duplicates based on these columns
    parsed = parsed.reset_index(drop=True).drop_duplicates(subset=subset_columns)
    parsed['item_category'] = pd.to_numeric(parsed['item_category'], errors='coerce')
    parsed['item_category'] = parsed['item_category'].fillna(-1).astype(int)
    parsed = pd.concat([parsed, phones], ignore_index=True )
    return parsed















def safe_eval(x):
    """Safely converts a string representation of a dictionary to an actual dictionary."""
    if pd.isna(x) or not isinstance(x, str) or x.strip() in ["", "None", "null"]:  
        return {}  # Return empty dict for missing or invalid data
    try:
        return ast.literal_eval(x)  # Try parsing as Python dict
    except (ValueError, SyntaxError):
        try:
            return json.loads(x.replace("'", "\""))  # Try JSON decoding
        except json.JSONDecodeError:
            return {}  # Return empty dict if all parsing fails


def normalize_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[+/()]', ' ', text)  # Replace special characters with spaces
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = unicodedata.normalize('NFKD', text)
        return ''.join(c for c in text if not unicodedata.combining(c)).strip()
    return ''

def parsed_brand_detection(parsed, umico):
    parsed = parsed[(parsed['item_category'].notna()) & (parsed['item_category'] != -1)]
    parsed['item_specs'] = parsed['item_specs'].astype(str)
    parsed['item_specs'] = parsed['item_specs'].str.lower()
    # Convert strings to dictionaries
    parsed['item_specs'] = parsed['item_specs'].apply(safe_eval)

    # Expand dictionary into separate columns
    df_expanded = parsed['item_specs'].apply(pd.Series)

    # Merge back and drop original column
    parsed = pd.concat([parsed, df_expanded], axis=1).drop(columns=['item_specs'])
    parsed.replace('nan', np.nan, inplace=True)
    parsed = parsed.dropna(subset=['item_name'])


    parsed['item_name'] = parsed['item_name'].str.lower().str.strip()
    umico = umico.rename({'Manufacturer': 'brand'}, axis=1)
    umico[['ProductName', 'brand']] = umico[['ProductName', 'brand']].apply(lambda x: x.str.lower())
    parsed = parsed.dropna(subset=['item_category'])

    # Define brand mappings as regex patterns for replacement
    brand_mappings = {
        r'(?<!\w)iphone(?!\w)': 'apple',
        r'(?<!\w)macbook(?!\w)': 'apple',
        r'(?<!\w)ipad(?!\w)': 'apple',
        r'(?<!\w)i̇pad(?!\w)': 'apple',
        r'(?<!\w)airpods(?!\w)': 'apple',
        r'(?<!\w)amazifit(?!\w)': 'xiaomi',
        r'(?<!\w)redmi(?!\w)': 'xiaomi',
        r'(?<!\w)galaxy(?!\w)': 'samsung',
        r'(?<!\w)thinkpad(?!\w)': 'lenovo',
        r'(?<!\w)playstation(?!\w)': 'sony',
        r'(?<!\w)xiaomi(?!\w)': 'xiaomi',
    }

    # Apply regex brand mapping
    for pattern, brand in brand_mappings.items():
        parsed.loc[parsed['item_name'].str.contains(pattern, regex=True, na=False), 'brand'] = brand

    # Create category-brand dictionary
    category_brand_dict = {
        cat: sorted(
            [b.lower() for b in brands if b and b.lower() != 'smart watch'],  # Convert brands to lowercase
            key=len, 
            reverse=True
        )
        for cat, brands in umico.groupby('CategoryID')['brand'].unique().items()
    }

    def extract_brand(row):
        cat_id = row['item_category']
        item_name = normalize_text(row['item_name'])

        if cat_id in category_brand_dict:
            brand_pattern = (
                r'(?<![A-Za-zА-Яа-яЁё0-9])(' + '|'.join(map(re.escape, category_brand_dict[cat_id])) + r')(?![A-Za-zА-Яа-яЁё0-9])' +
                r'|' +
                r'\b(' + '|'.join(map(re.escape, category_brand_dict[cat_id])) + r')\b'
            )
            match = re.search(brand_pattern, item_name)
            if match:
                return match.group(0)

        # Fallback: Check brand_mappings
        for pattern, brand in brand_mappings.items():
            if re.search(pattern, item_name):
                return brand

        return np.nan

    # Apply brand extraction
    parsed['brand'] = parsed.apply(extract_brand, axis=1)

    parsed = parsed.reset_index(drop=True)
    parsed = parsed[~parsed['item_price'].isna()]
    parsed = parsed.drop_duplicates(['item_url', 'item_category'], keep='first')
    parsed['item_category'] = parsed['item_category'].astype(int)

    parsed = parsed[sorted(parsed.columns)]

    # Apply condition and update 'item_name' for rows where 'item_category' is 3 and 'brand' is 'apple'
    # parsed.loc[(parsed['item_category'] == 3) & (parsed['brand'] == 'apple'), 'item_name'] = \
    #     parsed['item_name'] + ' ' + parsed['ram'].fillna('')
    
    return parsed
