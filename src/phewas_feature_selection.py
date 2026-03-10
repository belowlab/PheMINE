import pandas as pd
import polars as pl
import numpy as np
import os
import argparse
import sys
import logging
import time
from pathlib import Path
import warnings
import yaml
from phetk.phewas import PheWAS

warnings.filterwarnings('ignore')

try:
    with open("../config.yaml") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

def setup_log(fn_log, mode='w'):
    """Print log message to console and write to a log file."""
    logging.root.handlers = []
    logging.basicConfig(level=logging.DEBUG,
                        handlers=[logging.FileHandler(filename=fn_log, mode=mode),
                                  logging.StreamHandler()], format='%(message)s')

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_fn', help='List of cases', type=str,
                        default='./sample_data/case_list.txt')
    parser.add_argument('--control_fn', help='Table of matched controls of each case', type=str,
                        default='./sample_data/control_list.txt')
    parser.add_argument('--output_path', type=str, default='./')
    parser.add_argument('--output_prefix', type=str, default='output')
    parser.add_argument('--trait', help='Trait of interest', type=str, default='trait')
    parser.add_argument('--control_delimiter', default='tab', choices=[',', 'tab', 'space', 'whitespace'],
                        help='Delimiter of the control file')
    parser.add_argument('--phecode_binary_feather_file', type=str, default=config.get('phecode_binary_feather_file', ''),
                        help='Path to the feather file with binary phecode data')
    parser.add_argument('--demographics_file', type=str, default=config.get('demographics_file', ''),
                        help='Path to the demographics file for sex_at_birth')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
    # Record script used
    fn_log = os.path.join(args.output_path, args.output_prefix+'_phewas_feature_sel.log')
    setup_log(fn_log, mode='w')

    cmd_used = 'python ' + ' '.join(sys.argv)
    logging.info('\n# Call used:')
    logging.info(cmd_used+'\n')
    
    dict_delimiter = {',':',', 'tab':'\t', 'space':' ', 'whitespace':r'\s+'}
    if not args.control_delimiter:
        if args.control_fn.endswith('.csv'):
            args.control_delimiter = ','
        else:
            args.control_delimiter = '\t'
    else:
        args.control_delimiter = dict_delimiter[args.control_delimiter]
    return args

def get_case_control_lists(in_fn, delimiter):
    """Returns list of unique cases and unique controls from matching file."""
    df = pd.read_csv(in_fn, sep=delimiter)
    cases = df['case'].dropna().unique().tolist()
    
    control_cols = [col for col in df.columns if col.startswith('Control')]
    controls = pd.unique(df[control_cols].values.ravel('K'))
    controls = [c for c in controls if pd.notna(c)]
    
    return cases, list(set(controls))

def main():
    args = process_args()
    start_time = time.time()
    
    logging.info('\n# 1. Loading configuration and lists of cases/controls')
    cases, controls = get_case_control_lists(args.control_fn, args.control_delimiter)
    all_subjects = list(set(cases + controls))
    logging.info(f'Loaded {len(cases)} cases and {len(controls)} controls ({len(all_subjects)} total subjects)')

    # Cohort file generation
    logging.info('\n# 2. Generating Cohort file for PheTK')
    # Create basic cohort df
    cohort_df = pd.DataFrame({'grid': all_subjects})
    cohort_df['label'] = cohort_df['grid'].isin(cases).astype(int)
    
    # Needs sex/gender information (1=male, 0=female for PheTK)
    logging.info(f'Loading demographics from {args.demographics_file}')
    if str(args.demographics_file).endswith('.csv.gz') or str(args.demographics_file).endswith('.csv'):
        demo_df = pd.read_csv(args.demographics_file, usecols=['grid', 'gender_source_value'])
    else:
        demo_df = pd.read_csv(args.demographics_file, sep='\t', usecols=['grid', 'gender_source_value'])
        
    # Merge demographics
    cohort_df = cohort_df.merge(demo_df, on='grid', how='left')
    
    # Map sex to 0/1 (M=1, F=0)
    # Assuming 'M' and 'F'. Adjust if different.
    sex_mapping = {'M': 1, 'F': 0, 'Male': 1, 'Female': 0, '1': 1, '0': 0, 1: 1, 0: 0}
    cohort_df['sex'] = cohort_df['gender_source_value'].map(sex_mapping)
    
    # Drop records with missing sex if any
    missing_sex = cohort_df['sex'].isna().sum()
    if missing_sex > 0:
        logging.warning(f'Dropping {missing_sex} records with missing sex information.')
        cohort_df = cohort_df.dropna(subset=['sex'])
    
    cohort_df['sex'] = cohort_df['sex'].astype(int)
    
    # Rename grid to person_id for PheWAS
    # Write subset and merge with demographics file to get gender_source_value and age_in_days
    cohort_file_path = os.path.join(args.output_path, f"{args.output_prefix}_phewas_cohort.tsv")
    cohort_df.rename(columns={'grid': 'person_id'}, inplace=True) # Renamed grid to person_id earlier
    cohort_df[['person_id', 'label', 'sex']].to_csv(cohort_file_path, sep='\t', index=False)
    logging.info(f'Saved cohort file to {cohort_file_path}')

    # Phecode counts file generation
    logging.info('\n# 3. Generating Phecode Counts file for PheTK')
    logging.info(f"Loading binary phecodes from {args.phecode_binary_feather_file} (lazy)")
    df_phecode_lazy = pl.scan_ipc(args.phecode_binary_feather_file)
    
    # Filter for relevant grids
    relevant_grids = cohort_df['person_id'].tolist()
    logging.info(f"Collecting binary phecode data for {len(relevant_grids)} subjects in cohort...")
    subset_df = df_phecode_lazy.filter(pl.col('grid').is_in(relevant_grids)).collect().to_pandas()
    
    # Rename grid to person_id for PheWAS
    subset_df.rename(columns={'grid': 'person_id'}, inplace=True)
    
    # Convert to long format (person_id, phecode, count)
    long_df = subset_df.melt(id_vars=['person_id'], var_name='phecode', value_name='count')
    # Keep only non-zero counts
    long_df = long_df[long_df['count'] > 0]
    
    counts_file_path = os.path.join(args.output_path, f"{args.output_prefix}_phewas_counts.tsv")
    long_df.to_csv(counts_file_path, sep='\t', index=False)
    logging.info(f'Saved phecode counts file to {counts_file_path}')
    
    logging.info('\n# 4. Running PheTK PheWAS')
    phewas_output_path = os.path.join(args.output_path, f"{args.output_prefix}_phewas_results.tsv")
    
    # Run PheWAS for all phecodes predicting case/control status (label)
    try:
        phewas = PheWAS(
            phecode_version="1.2", # assuming standard phecodes
            phecode_count_file_path=counts_file_path,
            cohort_file_path=cohort_file_path,
            covariate_cols=["sex"],
            independent_variable_of_interest="label",
            sex_at_birth_col="sex",
            male_as_one=True,
            min_cases=round(cohort_df['label'].sum() * config.get('min_phecode_frequency', 0.02)), # dynamically match config
            min_phecode_count=1,
            output_file_path=phewas_output_path,
            verbose=False
        )
        phewas.run()
        logging.info(f"PheWAS completed. Results saved to {phewas_output_path}")
    except Exception as e:
        logging.error(f"Error running PheTK PheWAS: {e}")
        sys.exit(1)

    logging.info('\n# 5. Formatting Output for Pipeline')
    results_df = pd.read_csv(phewas_output_path, sep='\t', dtype={'phecode': str})
    
    # Filter significant results (e.g. p_value < 1e-5)
    sig_cutoff = 1e-5
    sig_results = results_df[results_df['p_value'] < sig_cutoff].copy()
    
    # Match the column names expected by pheML_develop and generate_reports.
    # The pipeline expects [TRAIT]_[prefix]_enriched_phecode.csv with columns:
    # Phecode, Description, Count, p.value
    
    sig_results.rename(columns={
        'phecode': 'Phecode',
        'phecode_string': 'Description',
        'p_value': 'p.value',
        'cases': 'Count' # It's case counts
    }, inplace=True)
    
    # Ensure count is just cases with the phecode (n_cases returned by PheTK is the number of cohort cases that have the phecode)
    # Actually PheTK returns `n_cases` as total number of individuals with the phecode if independent variable is continuous, or case/control counts if binary
    
    out_table = sig_results[['Phecode', 'Description', 'Count', 'p.value']].copy()
    sig_results = sig_results[['Phecode', 'Description', 'Count', 'p.value']]
    
    final_output_path = os.path.join(args.output_path, f"{args.trait}_{args.output_prefix}_phewas_enriched_phecode.csv")
    sig_results.to_csv(final_output_path, sep='\t', index=False)
    logging.info(f"Generated final feature selection file with {len(sig_results)} significant phecodes: {final_output_path}")
    
    time_elapsed = time.time() - start_time
    logging.info('\n# Done. Finished in %.2f minutes' % (time_elapsed/60))

if __name__ == '__main__':
    main()
