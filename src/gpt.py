import pandas as pd
import re
import os
from openai import AzureOpenAI
from loguru import logger
import concurrent.futures
import time

def gpt_prep(name):
        name = re.sub(r'\(.*?\)', '', name).strip().lower()
        replacements = {
        ',': ' ',
        '/': ' ',
        '.': ' ',
        '`':'',
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
        'gb':' ',
        '4g': ' ',
        '5g': ' ',
        'ё': 'е',
        'nfc': ' ',
        '   ': ' ',
        '  ':' '
    }
        for x, y in replacements.items():
                name = name.replace(x, y)

        return name.strip().lower()
    

def get_client(endpoint: str, key: str, api_version):
    os.environ["OPENAI_API_TYPE"] = "azure"
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=key)
    return client



def check_similarity(product_name, neighbor_name):
      
    api_version = os.environ['OPENAI_API_VERSION']
    key = os.environ['OPENAI_API_KEY']
    endpoint = os.environ['OPENAI_API_ENDPOINT']
    deployment_name = os.environ['OPENAI_DEPLOYMENT_NAME']
    client = get_client(endpoint, key, api_version)
    
    time.sleep(1)

    prompt = f"""
You are a product matching specialist. Your task is to determine if the two product names refer to the same product.

**Instructions:**
- Compare the **Primary Product Name**: '{product_name}' and the **Candidate Product Name**: '{neighbor_name}'.
- Consider if they are essentially the same in real life, taking into account their characteristics and color.
- Ignore country labels like "US," "RU," "RU3" when making your decision.
- Be aware that color names might be in different languages or have variations.
- Ensure that the ProductName and NeighborName in your output are exactly the same as in the input. Do not alter or translate the names.

**Output**: Answer with "True" if they refer to the same product, or "False" if they do not.
- Do not include any additional text or explanations.
- Matches must be based on meaning. For example, if two products have identical characteristics, color (even in different languages), and purpose, they are considered the same.
"""

    # API call using ChatCompletion
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=5,  # Limit tokens since we expect only a True/False response
        temperature=0.0  # Low temperature for more consistent results
    )

    result = response.choices[0].message.content.strip()
    logger.info(f"'{product_name}' and '{neighbor_name}' Match '{result}'")


    return result





# Parallel processing function with incremental saving to CSV
def process_in_parallel(df, func, output_file,  num_workers=4):

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Map rows to tuples of (item_name_cln, ProductName_cln)
        rows = [(row['product'], row['neighbor']) for _, row in df.iterrows()]
        
        # Open the CSV file for appending results
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            for idx, (product, neighbor) in enumerate(rows):
                match_result = func(neighbor, product)
                # Prepare the result for this iteration
                result_data = {
                    'product': product,
                    'neighbor': neighbor,
                    'Match': match_result
                }
                # Append the result to the CSV file
                pd.DataFrame([result_data]).to_csv(f, header=f.tell() == 0, index=False)  # Write header only once

    print(f"Results have been appended to {output_file}")








def process_and_match_products(candidates, output_file, n_cash):

    # candidates['neighbor_cln'] = candidates['neighbor'].apply(gpt_prep)
    # candidates['product_cln'] = candidates['product'].apply(gpt_prep)
    candidates['sorted_tuple'] = candidates.apply(lambda row: tuple(sorted([row['neighbor'], row['product']])), axis=1)
    pairs = candidates[['sorted_tuple']].drop_duplicates()

    gpt_output = pd.read_csv(output_file)
    gpt_output = gpt_output[n_cash:]
    gpt_output['Pair'] = gpt_output.apply(lambda row: tuple(sorted([row['product'], row['neighbor']])), axis=1)

    pairs = set(pairs['sorted_tuple']) - set(gpt_output['Pair'])
    print(f'Pairs to compare {len(pairs)}')
    pairs = pd.DataFrame(list(pairs), columns=['neighbor', 'product'])

    process_in_parallel(pairs, check_similarity, num_workers=1, output_file=output_file)

    gpt_output = pd.read_csv(output_file).drop_duplicates()
    gpt_output.to_csv(output_file, index=False)
    gpt_output['Pair'] = gpt_output.apply(lambda row: tuple(sorted([row['product'], row['neighbor']])), axis=1)

    candidates = candidates.merge(gpt_output[['Match', 'Pair']], left_on='sorted_tuple', right_on='Pair', how='left')

    final_output = candidates[['product', 'neighbor', 'Match']]
    final_output['score'] = 100
    final_output.columns = ['product', 'neighbor', 'predicted_match', 'score']

    return final_output