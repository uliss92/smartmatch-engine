
import pandas as pd
from utils import upload_data
import argparse
from pathlib import Path
from script import process_and_generate_candidates, load_model_and_predict, process_product_matching,rapidfuzz_cutoff, concat_last_n_days_data
import os
import json
from datetime import datetime
from gpt import process_and_match_products
from parsed_prep import map_categories_to_parsed_products, parsed_brand_detection
import time
import csv



today_date = datetime.today().strftime('%d.%m.%Y')

parser = argparse.ArgumentParser()

 
parser.add_argument('--root', type=str)
args = parser.parse_args()
ds_root = args.root



# Load data
# golden = pd.read_csv(Path(ds_root) / "/fine_tuning/final.csv")



# Read the JSON file
with open(Path(ds_root) / "input" / "params.json", 'r', encoding='utf-8') as file:
    params = json.load(file)

print("PARAMETERS")
print(json.dumps(params, indent=4, ensure_ascii=False))


default_threshold = params.get('default_threshold')
labse_threshold = params.get('labse_threshold')
fuzzy_threshold = params.get('fuzzy_threshold')
debug_cats = params.get('debug_cats')
fuzzy_on_off = params.get('fuzzy_on_off')
# n_days = params.get('n_days')
n_cash = params.get('n_cash')



parsed = pd.read_csv(Path(ds_root)  / "input" / "parsed_items-modified.csv")
main_market = pd.read_parquet(Path(ds_root) / "input" / "mp_products_mpn_gtin.parquet")
chars = pd.read_parquet(Path(ds_root) / "input" / "mp_price_comparision_all_products_chars.parquet")
offers = pd.read_parquet(Path(ds_root)  / "input" / "mp_all_offers_prod.parquet")
category_tree_with_translation = pd.read_csv(Path(ds_root)  / "input" /'category_tree_with_translation.csv')
stopwords = pd.read_excel(Path(ds_root)  / "input" /'stop_words.xlsx')['word']


output_file = Path(ds_root) / 'output/GPT_dups_final.csv'
output_file = output_file.resolve().as_posix()




tokenizer_path = Path(ds_root) / 'fine_tuning/fine_tuned/fine_tuned_tokenizer' 
tokenizer_path = tokenizer_path.resolve().as_posix()

model_path = Path(ds_root) / 'fine_tuning/fine_tuned/fine_tuned_model.pth' 
model_path = model_path.resolve().as_posix()

historical = Path(ds_root) / '/historical' 
historical = historical.resolve().as_posix()


csv_file = Path(ds_root) / 'output/product_category_mapping_GPT.csv'
csv_file = csv_file.as_posix()

# Check if the CSV exists, and if not, create it with headers
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Product', 'Category'])
        

        # Mapping of replacements
replacement_map = {
    'blender': 'blenderlər',
    'aspirator': 'aspiratorlar',
    'микроволновая печь': 'микроволновые печи',
    'kondisioner': 'kondisionerlər',
    'soyuducu': 'soyuducular',
    'tozsoran': 'adi tozsoranlar',
    'пылесос': 'пылесосы обычные',
    'пылесос -': 'пылесосы обычные',
    'пылесос-': 'пылесосы обычные'
}







def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) / 60
        print(f"\033[\033[{'=' * 50}")
        print(f"  {func.__name__.upper()} EXECUTED IN {execution_time:.2f} MINUTES")
        print(f"{'=' * 50}\033]")
        return result
    return wrapper

# Applying the decorator to functions
parsed = log_time(map_categories_to_parsed_products)(main_market, parsed, params, category_tree_with_translation, csv_file, replacement_map, n_cash)
parsed = log_time(parsed_brand_detection)(parsed, main_market)

parsed.to_csv("parsed.csv", index=False)

# Upload data
log_time(upload_data)(
    datastore_name="mp_ds_matching",
    files=["parsed.csv"],
    target_path="output/debug/"
)



if debug_cats: 
    parsed = parsed[parsed['item_category'].isin(debug_cats)]
    main_market = main_market[main_market['CategoryID'].isin(debug_cats)]
    offers = offers[offers['CategoryID'].isin(debug_cats)]
    chars = chars[chars['CategoryID'].isin(debug_cats)]

duplicates = main_market[main_market['CategoryID'] == 261].copy()
duplicates['CategoryID'] = 259
main_market.loc[main_market['CategoryID'] == 261, 'CategoryID'] = 260
main_market = pd.concat([main_market, duplicates], ignore_index=True)

duplicates = parsed[parsed['item_category'] == 261].copy()
duplicates['item_category'] = 259
parsed.loc[parsed['item_category'] == 261, 'item_category'] = 260
parsed = pd.concat([parsed, duplicates], ignore_index=True)

# Log time for candidate generation
main_market_n_parsed, candidates = log_time(process_and_generate_candidates)(params, main_market, chars, offers, parsed, default_threshold,category_tree_with_translation, stopwords)

main_market_n_parsed.to_csv("main_market_n_parsed.csv", index=False)
candidates.to_csv("candidates.csv", index=False)

# Upload data
log_time(upload_data)(
    datastore_name="mp_ds_matching",
    files=["main_market_n_parsed.csv", "candidates.csv"],
    target_path="output/debug/"
)

if fuzzy_on_off:
    candidates = log_time(rapidfuzz_cutoff)(candidates, threshold=fuzzy_threshold)
    candidate_pairs = log_time(process_and_match_products)(candidates, output_file, n_cash)
else:
    candidate_pairs = log_time(load_model_and_predict)(model_path, tokenizer_path, candidates, batch_size=16)



candidate_pairs.to_csv("candidate_pairs.csv", index=False)


# Upload data
log_time(upload_data)(
    datastore_name="mp_ds_matching",
    files=["candidate_pairs.csv"],
    target_path="output/debug/"
)



matched_df = log_time(process_product_matching)(candidate_pairs, main_market_n_parsed, threshold=labse_threshold)

matched_df = matched_df.drop_duplicates(keep='first')


matched_df.to_csv("matched_df.csv", index=False)
matched_df.to_parquet("matched_df.parquet", index=False)

# Upload data
log_time(upload_data)(
    datastore_name="mp_ds_matching",
    files=["matched_df.csv","matched_df.parquet"],
    target_path="output/"
)

# Historical data processing (if needed)
# matched_historical = log_time(concat_last_n_days_data)(historical, n_days)
# matched_historical.to_csv("matched_historical.csv", index=False)
# log_time(upload_data)(
#     datastore_name="mp_ds_matching",
#     files=["matched_historical.csv"],
#     target_path="output/"
# )


# parsed = map_categories_to_parsed_products(main_market, parsed, params, category_tree_with_translation, csv_file, replacement_map)
# parsed = parsed_brand_detection(parsed, chars, main_market, params)































# if debug_cats:  # Check if debug_cats is not empty
#     parsed = parsed[parsed['item_category'].isin(debug_cats)]
#     main_market = main_market[main_market['CategoryID'].isin(debug_cats)]
#     offers = offers[offers['CategoryID'].isin(debug_cats)]
#     chars = chars[chars['SrcCategoryID'].isin(debug_cats)]
# # mapping unisex perfume cats to mens and womens
# duplicates = main_market[main_market['CategoryID'] == 261].copy()
# duplicates['CategoryID'] = 259

# main_market.loc[main_market['CategoryID'] == 261, 'CategoryID'] = 260
# main_market = pd.concat([main_market, duplicates], ignore_index=True)

# duplicates = parsed[parsed['item_category'] == 261].copy()
# duplicates['item_category'] = 259

# parsed.loc[parsed['item_category'] == 261, 'item_category'] = 260
# parsed = pd.concat([parsed, duplicates], ignore_index=True)







# main_market_n_parsed, candidates = process_and_generate_candidates(params, main_market, chars, offers, parsed, default_threshold)




# if fuzzy_on_off:
#     candidates=rapidfuzz_cutoff(candidates, threshold=fuzzy_threshold)
#     candidate_pairs = process_and_match_products(candidates, output_file)
# else:
#     candidate_pairs = load_model_and_predict(model_path, tokenizer_path, candidates, batch_size=16)









# matched_df = process_product_matching(candidate_pairs, main_market_n_parsed, threshold=labse_threshold)




# main_market_n_parsed.to_csv("main_market_n_parsed.csv", index=False)
# candidates.to_csv("candidates.csv", index=False)
# candidate_pairs.to_csv("candidate_pairs.csv", index=False)
# # matched_df.to_csv(f"matched_df_{today_date}.csv", index=False)
# matched_df.to_csv("matched_df.csv", index=False)




# # uploading
# upload_data(
#     datastore_name="mp_ds_matching",
#     files=["main_market_n_parsed.csv", "candidates.csv", "candidate_pairs.csv","matched_df.csv" ],
#     target_path="output/"
# )


# # uploading
# # upload_data(
# #     datastore_name="mp_ds_matching",
# #     files=[f"matched_df_{today_date}.csv"],
# #     target_path="output/historical"
# # )


# # matched_historical = concat_last_n_days_data(historical, n_days)
# # matched_historical.to_csv(f"matched_historical.csv", index=False)


# # # uploading
# # upload_data(
# #     datastore_name="mp_ds_matching",
# #     files=["matched_historical.csv"],
# #     target_path="output/"
# # )