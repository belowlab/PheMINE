import argparse
import logging
import os
import sys
import time
import warnings

import pandas as pd
import polars as pl
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
    """Print log messages to console and a log file."""
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[logging.FileHandler(filename=fn_log, mode=mode), logging.StreamHandler()],
        format='%(message)s'
    )


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_fn', help='List of cases', type=str, default='./sample_data/case_list.txt')
    parser.add_argument('--control_fn', help='Table of matched controls of each case', type=str,
                        default='./sample_data/control_list.txt')
    parser.add_argument('--output_path', type=str, default='./')
    parser.add_argument('--output_prefix', type=str, default='output')
    parser.add_argument('--trait', help='Trait of interest', type=str, default='trait')
    parser.add_argument('--control_delimiter', default='tab', choices=[',', 'tab', 'space', 'whitespace'],
                        help='Delimiter of the control file')
    parser.add_argument('--phecode_binary_feather_file', type=str,
                        default=config.get('phecode_binary_feather_file', ''),
                        help='Path to the feather file with binary phecode data')
    parser.add_argument('--demographics_file', type=str, default=config.get('demographics_file', ''),
                        help='Path to the demographics file for sex_at_birth')
    parser.add_argument('--phecode_map_file', type=str, default=config.get('phecode_map_file', ''),
                        help='Path to the ICD to phecode mapping file')

    args = parser.parse_args()

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    fn_log = os.path.join(args.output_path, args.output_prefix + '_phewas_feature_sel.log')
    setup_log(fn_log, mode='w')

    cmd_used = 'python ' + ' '.join(sys.argv)
    logging.info('\n# Call used:')
    logging.info(cmd_used + '\n')

    delimiter_map = {',': ',', 'tab': '\t', 'space': ' ', 'whitespace': r'\s+'}
    args.control_delimiter = delimiter_map[args.control_delimiter]
    return args


def get_case_control_lists(in_fn, delimiter):
    """Return unique cases and controls from the matched-control file."""
    df = pd.read_csv(in_fn, sep=delimiter)
    cases = df['case'].dropna().astype(str).unique().tolist()

    control_cols = [col for col in df.columns if col.startswith('Control')]
    controls = pd.unique(df[control_cols].values.ravel('K'))
    controls = [str(c) for c in controls if pd.notna(c)]

    return cases, list(set(controls))


def load_phecode_map(phecode_map_file):
    """Load the ICD to phecode mapping file used to resolve excluded ICD codes."""
    if not phecode_map_file:
        return pd.DataFrame(columns=['ICD', 'Phecode'])

    phecode_map = pd.read_csv(phecode_map_file, dtype={'ICD': str, 'Phecode': str})
    keep_cols = [col for col in ['ICD', 'Phecode'] if col in phecode_map.columns]
    return phecode_map[keep_cols].copy()


def resolve_excluded_phecodes(phecode_map):
    """Resolve phenotype-defining phecodes that should be excluded from ML features."""
    excluded_codes = config.get('excluded_codes', {}) or {}
    excluded_phecodes = [str(code).strip() for code in (excluded_codes.get('Phecode', []) or []) if str(code).strip()]

    excluded_icd = excluded_codes.get('ICD', []) or []
    if excluded_icd and not phecode_map.empty and {'ICD', 'Phecode'}.issubset(phecode_map.columns):
        phecode_map_icd = phecode_map.copy()
        phecode_map_icd['ICD'] = phecode_map_icd['ICD'].astype(str).str.strip()
        phecode_map_icd['Phecode'] = phecode_map_icd['Phecode'].astype(str).str.strip()
        matched = phecode_map_icd[phecode_map_icd['ICD'].isin([str(code).strip() for code in excluded_icd])]
        excluded_phecodes.extend(matched['Phecode'].dropna().tolist())

    deduped = []
    for code in excluded_phecodes:
        if code and code not in deduped:
            deduped.append(code)
    return deduped


def load_demographics(demographics_file):
    """Load the sex field needed by PheTK."""
    logging.info(f'Loading demographics from {demographics_file}')
    if str(demographics_file).endswith('.csv.gz') or str(demographics_file).endswith('.csv'):
        return pd.read_csv(demographics_file, usecols=['grid', 'gender_source_value'])
    return pd.read_csv(demographics_file, sep='\t', usecols=['grid', 'gender_source_value'])


def build_cohort(cases, controls, demographics_file):
    """
    Build the matched cohort used for Phe-PheWAS.

    In this pipeline, the exposure phenotype is the case/control label itself.
    """
    all_subjects = list(set(cases + controls))
    cohort_df = pd.DataFrame({'grid': all_subjects})
    cohort_df['label'] = cohort_df['grid'].isin(set(cases)).astype(int)

    demo_df = load_demographics(demographics_file)
    demo_df['grid'] = demo_df['grid'].astype(str)
    cohort_df['grid'] = cohort_df['grid'].astype(str)
    cohort_df = cohort_df.merge(demo_df, on='grid', how='left')

    sex_mapping = {'M': 1, 'F': 0, 'Male': 1, 'Female': 0, '1': 1, '0': 0, 1: 1, 0: 0}
    cohort_df['sex'] = cohort_df['gender_source_value'].map(sex_mapping)

    missing_sex = cohort_df['sex'].isna().sum()
    if missing_sex > 0:
        logging.warning(f'Dropping {missing_sex} records with missing sex information.')
        cohort_df = cohort_df.dropna(subset=['sex'])

    cohort_df['sex'] = cohort_df['sex'].astype(int)
    logging.info(
        'Exposure phenotype prevalence in matched cohort: %d/%d cases',
        int(cohort_df['label'].sum()),
        cohort_df.shape[0]
    )
    return cohort_df


def load_binary_phecodes(feather_file, relevant_grids):
    """Load binary phecode data for the matched cohort."""
    logging.info(f"Loading binary phecodes from {feather_file} (lazy)")
    df_phecode_lazy = pl.scan_ipc(feather_file)
    logging.info(f"Collecting binary phecode data for {len(relevant_grids)} subjects in cohort...")
    subset_df = (
        df_phecode_lazy
        .filter(pl.col('grid').cast(pl.Utf8).is_in(relevant_grids))
        .collect()
        .to_pandas()
    )
    subset_df['grid'] = subset_df['grid'].astype(str)
    return subset_df


def write_phewas_inputs(cohort_df, subset_df, output_path, output_prefix):
    """Write the cohort and phecode count files consumed by PheTK."""
    cohort_file_path = os.path.join(output_path, f"{output_prefix}_phewas_cohort.tsv")
    cohort_out = cohort_df.rename(columns={'grid': 'person_id'}).copy()
    cohort_out[['person_id', 'label', 'sex']].to_csv(cohort_file_path, sep='\t', index=False)
    logging.info(f'Saved cohort file to {cohort_file_path}')

    counts_df = subset_df.rename(columns={'grid': 'person_id'}).copy()
    long_df = counts_df.melt(id_vars=['person_id'], var_name='phecode', value_name='count')
    long_df = long_df[long_df['count'] > 0]

    counts_file_path = os.path.join(output_path, f"{output_prefix}_phewas_counts.tsv")
    long_df.to_csv(counts_file_path, sep='\t', index=False)
    logging.info(f'Saved phecode counts file to {counts_file_path}')
    return cohort_file_path, counts_file_path


def run_phewas(cohort_df, cohort_file_path, counts_file_path, output_path, output_prefix):
    """Run a phenome scan using the matched case phenotype as the exposure."""
    phewas_output_path = os.path.join(output_path, f"{output_prefix}_phewas_results.tsv")
    min_cases = max(1, round(int(cohort_df['label'].sum()) * config.get('min_phecode_frequency', 0.02)))

    phewas = PheWAS(
        phecode_version="1.2",
        phecode_count_file_path=counts_file_path,
        cohort_file_path=cohort_file_path,
        covariate_cols=["sex"],
        independent_variable_of_interest="label",
        sex_at_birth_col="sex",
        male_as_one=True,
        min_cases=min_cases,
        min_phecode_count=1,
        output_file_path=phewas_output_path,
        verbose=False
    )
    phewas.run()
    logging.info(f'PheWAS completed. Results saved to {phewas_output_path}')
    return phewas_output_path


def format_feature_selection_output(results_file, output_path, trait, output_prefix, excluded_phecodes):
    """Format PheTK output into the feature-selection table consumed by the ML step."""
    results_df = pd.read_csv(results_file, sep='\t', dtype={'phecode': str})

    if 'p_value' not in results_df.columns:
        raise ValueError('PheTK output is missing the p_value column required for filtering.')

    selected_results = results_df.copy()
    if excluded_phecodes:
        selected_results = selected_results[
            ~selected_results['phecode'].astype(str).isin(set(excluded_phecodes))
        ]

    sig_cutoff = config.get('phewas_sig_cutoff', 1e-5)
    formatted = selected_results[selected_results['p_value'] < sig_cutoff].copy()
    if formatted.empty:
        logging.warning(
            'No phecodes met the significance cutoff. Falling back to all tested phecodes ranked by p-value.'
        )
        formatted = selected_results.copy()

    description_col = 'phecode_string' if 'phecode_string' in formatted.columns else 'PhecodeString'
    count_col = None
    for candidate in ['cases', 'n_cases', 'positive_observations', 'count']:
        if candidate in formatted.columns:
            count_col = candidate
            break
    if count_col is None:
        raise ValueError('Unable to find a count column in PheTK output.')

    rename_map = {
        'phecode': 'Phecode',
        description_col: 'Description',
        'p_value': 'p.value',
        count_col: 'Count'
    }
    formatted.rename(columns=rename_map, inplace=True)

    for required_col in ['Phecode', 'Description', 'Count', 'p.value']:
        if required_col not in formatted.columns:
            raise ValueError(f'Missing required output column after formatting: {required_col}')

    formatted = formatted[['Phecode', 'Description', 'Count', 'p.value']].copy()
    formatted.sort_values(by=['p.value', 'Count'], ascending=[True, False], inplace=True)

    final_output_path = os.path.join(output_path, f"{trait}_{output_prefix}_phewas_enriched_phecode.csv")
    formatted.to_csv(final_output_path, sep='\t', index=False)
    logging.info(
        'Generated final feature selection file with %d phecodes: %s',
        len(formatted),
        final_output_path
    )
    return final_output_path


def main():
    args = process_args()
    start_time = time.time()

    logging.info('\n# 1. Loading configuration and matched cohort')
    cases, controls = get_case_control_lists(args.control_fn, args.control_delimiter)
    cohort_df = build_cohort(cases, controls, args.demographics_file)
    logging.info(
        'Loaded %d cases and %d controls (%d total subjects after sex filtering)',
        len(cases),
        len(controls),
        cohort_df.shape[0]
    )

    logging.info('\n# 2. Resolving excluded phecodes for leakage control')
    phecode_map = load_phecode_map(args.phecode_map_file)
    excluded_phecodes = resolve_excluded_phecodes(phecode_map)

    logging.info('\n# 3. Loading cohort phecodes')
    subset_df = load_binary_phecodes(args.phecode_binary_feather_file, cohort_df['grid'].astype(str).tolist())
    subset_df = subset_df[subset_df['grid'].isin(cohort_df['grid'])].copy()
    cohort_df = cohort_df[cohort_df['grid'].isin(subset_df['grid'])].copy()

    logging.info('\n# 4. Writing PheTK input files')
    cohort_file_path, counts_file_path = write_phewas_inputs(
        cohort_df=cohort_df,
        subset_df=subset_df,
        output_path=args.output_path,
        output_prefix=args.output_prefix
    )

    logging.info('\n# 5. Running phenotype-to-phenome scan')
    try:
        phewas_output_path = run_phewas(
            cohort_df=cohort_df,
            cohort_file_path=cohort_file_path,
            counts_file_path=counts_file_path,
            output_path=args.output_path,
            output_prefix=args.output_prefix
        )
    except Exception as e:
        logging.error(f'Error running PheTK PheWAS: {e}')
        sys.exit(1)

    logging.info('\n# 6. Formatting PheTK output for ML feature selection')
    format_feature_selection_output(
        results_file=phewas_output_path,
        output_path=args.output_path,
        trait=args.trait,
        output_prefix=args.output_prefix,
        excluded_phecodes=excluded_phecodes
    )

    time_elapsed = time.time() - start_time
    logging.info('\n# Done. Finished in %.2f minutes', time_elapsed / 60)


if __name__ == '__main__':
    main()
