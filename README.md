# SmartMatch Engine

SmartMatch Engine automates product matching across large marketplace catalogues. It combines rule-based preprocessing, transformer-based embeddings, and GPT-assisted review to identify equivalent products with high precision while reducing the need for manual curation.

## Features
- **Scalable data ingestion** – Normalizes catalogues, offer data, and attribute feeds from heterogeneous sources.
- **Adaptive category alignment** – Automatically reconciles category trees using configurable thresholds and GPT-assisted fallbacks.
- **Hybrid candidate generation** – Mixes deterministic filtering, RapidFuzz scoring, and LaBSE similarity for robust shortlist creation.
- **Human-in-the-loop ready** – Produces audit artefacts (CSV/Parquet) and debug views to integrate with analyst review workflows.

## High-Level Workflow
1. Load marketplace catalogues, attribute feeds, and historical offer data.
2. Harmonize taxonomies and enrich attributes using configurable mappings defined in `params.json`.
3. Generate match candidates through heuristic filtering and string similarity scoring.
4. Apply neural ranking (fine-tuned LaBSE) and optional GPT validation for final decisions.
5. Persist grouped matches and diagnostics for downstream systems.

## Getting Started
### Prerequisites
- Python 3.10+
- Access to Azure resources (for data uploads and optional GPT checks)
- Dataset folder structured as:
  ```
  <ds_root>/
    input/
      parsed_items-modified.csv
      mp_products_mpn_gtin.parquet
      mp_price_comparision_all_products_chars.parquet
      mp_all_offers_prod.parquet
      category_tree_with_translation.csv
      stop_words.xlsx
    output/
    fine_tuning/
  ```

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
az login
python -m build
```

### Configuration
`params.json` controls thresholds and taxonomy mappings. A starter template is provided at the project root. Copy it to `<ds_root>/input/params.json` and customize the values for your catalogue:
```json
{
  "default_threshold": 0.72,
  "labse_threshold": 0.80,
  "fuzzy_threshold": 85,
  "debug_cats": [],
  "fuzzy_on_off": true,
  "n_cash": 100,
  "category": {
    "3": 0.82,
    "10": 0.75,
    "259": 0.78,
    "260": 0.78
  },
  "chars_mapping": {
    "ram": "ram",
    "rom": "rom",
    "volume": "volume",
    "colour": "color",
    "screen_size": "screen_size"
  }
}
```
Adjust category thresholds, add attribute mappings, and set `debug_cats` to target specific category IDs during experimentation.

### Run the Pipeline
```bash
python src/main.py --root <ds_root>
```
This command orchestrates preprocessing, candidate generation, RapidFuzz filtering, and either GPT or LaBSE confirmation. Outputs are written to `<ds_root>/output/` and mirrored to the configured Azure datastore for auditing.

## Project Structure
```
.
├── README.md
├── params.json                # Sample configuration
├── src/
│   ├── main.py                # Orchestrates the end-to-end pipeline
│   ├── script.py              # Candidate generation, scoring, and grouping utilities
│   ├── parsed_prep.py         # Category harmonization and attribute enrichment
│   └── gpt.py                 # GPT-assisted validation helpers
├── fine_tuning_labse/         # Experiments and assets for LaBSE fine-tuning
├── requirements.txt           # Python dependencies
├── conda_dependencies.yml     # Alternative environment specification
└── utils.py                   # Shared Azure storage helpers
```

## Career Materials
Looking to present this project in professional channels? See [`docs/professional_snippets.txt`](docs/professional_snippets.txt) for ready-to-use resume and LinkedIn project descriptions.

## Contributing
1. Fork the repository and create a feature branch.
2. Ensure new code is linted and covered by tests where applicable.
3. Submit a pull request detailing the problem solved and validation performed.

## License
This project is provided without a specific license. Contact the maintainers to discuss usage rights for commercial projects.
