import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

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
    """Print log message to console and write to a log file."""
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

    dict_delimiter = {',': ',', 'tab': '\t', 'space': ' ', 'whitespace': r'\s+'}
    if not args.control_delimiter:
        if args.control_fn.endswith('.csv'):
            args.control_delimiter = ','
        else:
            args.control_delimiter = '\t'
    else:
        args.control_delimiter = dict_delimiter[args.control_delimiter]
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


def resolve_exposure_phecodes(phecode_map):
    """
    Resolve the phenotype used as the Phe-PheWAS exposure.

    Priority:
    1. config['phewas_exposure_phecodes']
    2. config['excluded_codes']['Phecode']
    3. config['excluded_codes']['ICD'] mapped to phecodes
    """
    exposure_phecodes = []

    configured = config.get('phewas_exposure_phecodes', []) or []
    exposure_phecodes.extend(str(code).strip() for code in configured if str(code).strip())

    excluded_codes = config.get('excluded_codes', {}) or {}
    excluded_phecodes = excluded_codes.get('Phecode', []) or []
    exposure_phecodes.extend(str(code).strip() for code in excluded_phecodes if str(code).strip())

    excluded_icd = excluded_codes.get('ICD', []) or []
    if excluded_icd and not phecode_map.empty and {'ICD', 'Phecode'}.issubset(phecode_map.columns):
        phecode_map_icd = phecode_map.copy()
        phecode_map_icd['ICD'] = phecode_map_icd['ICD'].astype(str).str.strip()
        phecode_map_icd['Phecode'] = phecode_map_icd['Phecode'].astype(str).str.strip()
        matched = phecode_map_icd[phecode_map_icd['ICD'].isin([str(code).strip() for code in excluded_icd])]
        exposure_phecodes.extend(matched['Phecode'].dropna().tolist())

    deduped = []
    for code in exposure_phecodes:
        if code and code not in deduped:
            deduped.append(code)
    return deduped


def load_demographics(demographics_file):
    """Load demographics with the sex field needed by PheTK."""
    logging.info(f'Loading demographics from {demographics_file}')
    if str(demographics_file).endswith('.csv.gz') or str(demographics_file).endswith('.csv'):
        return pd.read_csv(demographics_file, usecols=['grid', 'gender_source_value'])
    return pd.read_csv(demographics_file, sep='\t', usecols=['grid', 'gender_source_value'])


def build_cohort(cases, controls, demographics_file):
    """Build the matched cohort used for feature selection."""
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
    return cohort_df


def load_binary_phecodes(feather_file, relevant_grids):
    """Load binary phecode data for the cohort."""
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


def resolve_available_exposure_phecodes(subset_df, exposure_phecodes):
    """Keep only configured exposure phecodes that exist in the cohort matrix."""
    available_phecodes = [code for code in exposure_phecodes if code in subset_df.columns]
    missing_phecodes = [code for code in exposure_phecodes if code not in subset_df.columns]

    if missing_phecodes:
        logging.warning(
            'The following exposure phecodes were not found in the binary phecode file: %s',
            ', '.join(missing_phecodes)
        )

    return available_phecodes


def build_exposure_cohort(cohort_df, subset_df, exposure_phecode=None):
    """
    Create the cohort file for a single exposure phecode.

    If exposure_phecode is None, fall back to the matched case/control label so
    the pipeline remains runnable.
    """
    cohort_out = cohort_df.copy()

    if exposure_phecode is None:
        cohort_out['exposure'] = cohort_out['label']
        analysis_mode = 'case_control_fallback'
        exposure_label = 'case_control_label'
    else:
        exposure_df = subset_df[['grid', exposure_phecode]].copy()
        exposure_df.rename(columns={exposure_phecode: 'exposure'}, inplace=True)
        cohort_out = cohort_out.merge(exposure_df, on='grid', how='left')
        cohort_out['exposure'] = cohort_out['exposure'].fillna(0).astype(int)
        analysis_mode = 'phe_phewas'
        exposure_label = exposure_phecode

    exposure_cases = int(cohort_out['exposure'].sum())
    logging.info(
        'Exposure %s prevalence in cohort: %d/%d subjects',
        exposure_label,
        exposure_cases,
        cohort_out.shape[0]
    )
    return cohort_out, exposure_label, analysis_mode


def write_cohort_file(cohort_df, output_path, output_prefix, exposure_label):
    """Write the cohort file for a single exposure-specific PheTK run."""
    safe_exposure = str(exposure_label).replace('/', '_').replace(' ', '_')
    cohort_file_path = os.path.join(output_path, f"{output_prefix}_phewas_cohort_{safe_exposure}.tsv")
    cohort_out = cohort_df.rename(columns={'grid': 'person_id'}).copy()
    cohort_out[['person_id', 'label', 'exposure', 'sex']].to_csv(cohort_file_path, sep='\t', index=False)
    logging.info(f'Saved cohort file to {cohort_file_path}')
    return cohort_file_path


def write_counts_file(subset_df, output_path, output_prefix):
    """Write the shared phecode count file consumed by PheTK."""
    counts_df = subset_df.rename(columns={'grid': 'person_id'}).copy()
    long_df = counts_df.melt(id_vars=['person_id'], var_name='phecode', value_name='count')
    long_df = long_df[long_df['count'] > 0]

    counts_file_path = os.path.join(output_path, f"{output_prefix}_phewas_counts.tsv")
    long_df.to_csv(counts_file_path, sep='\t', index=False)
    logging.info(f'Saved phecode counts file to {counts_file_path}')
    return counts_file_path


def run_phewas(cohort_df, cohort_file_path, counts_file_path, output_path, output_prefix, exposure_label):
    """Run one PheTK scan for a single exposure phenotype."""
    safe_exposure = str(exposure_label).replace('/', '_').replace(' ', '_')
    phewas_output_path = os.path.join(output_path, f"{output_prefix}_phewas_results_{safe_exposure}.tsv")
    exposure_count = max(int(cohort_df['exposure'].sum()), 1)
    min_cases = max(1, round(exposure_count * config.get('min_phecode_frequency', 0.02)))

    phewas = PheWAS(
        phecode_version="1.2",
        phecode_count_file_path=counts_file_path,
        cohort_file_path=cohort_file_path,
        covariate_cols=["sex"],
        independent_variable_of_interest="exposure",
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


def format_single_phewas_output(results_file, excluded_phecodes, exposure_label):
    """Format one exposure-specific PheTK result table."""
    results_df = pd.read_csv(results_file, sep='\t', dtype={'phecode': str})

    sig_cutoff = config.get('phewas_sig_cutoff', 1e-5)
    if 'p_value' not in results_df.columns:
        raise ValueError('PheTK output is missing the p_value column required for filtering.')

    selected_results = results_df.copy()
    if excluded_phecodes:
        selected_results = selected_results[
            ~selected_results['phecode'].astype(str).isin(set(excluded_phecodes))
        ]
    sig_results = selected_results[selected_results['p_value'] < sig_cutoff].copy()
    if sig_results.empty:
        logging.warning(
            'No phecodes met the significance cutoff for exposure %s. '
            'Falling back to all tested phecodes ranked by p-value.',
            exposure_label
        )
        sig_results = selected_results.copy()

    description_col = 'phecode_string' if 'phecode_string' in sig_results.columns else 'PhecodeString'
    count_col = None
    for candidate in ['cases', 'n_cases', 'positive_observations', 'count']:
        if candidate in sig_results.columns:
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
    sig_results.rename(columns=rename_map, inplace=True)

    for required_col in ['Phecode', 'Description', 'Count', 'p.value']:
        if required_col not in sig_results.columns:
            raise ValueError(f'Missing required output column after formatting: {required_col}')

    formatted = sig_results[['Phecode', 'Description', 'Count', 'p.value']].copy()
    formatted['ExposurePhecode'] = exposure_label
    return formatted


def combine_feature_selection_outputs(formatted_results, output_path, trait, output_prefix):
    """Combine exposure-specific Phe-PheWAS hits into one feature-selection table."""
    if not formatted_results:
        formatted = pd.DataFrame(columns=['Phecode', 'Description', 'Count', 'p.value', 'ExposurePhecode'])
    else:
        combined = pd.concat(formatted_results, ignore_index=True)
        combined.sort_values(by=['p.value', 'Count'], ascending=[True, False], inplace=True)
        exposure_map = (
            combined.groupby('Phecode')['ExposurePhecode']
            .apply(lambda values: ','.join(sorted(set(values))))
            .rename('ExposurePhecode')
        )
        formatted = (
            combined
            .drop_duplicates(subset=['Phecode'], keep='first')
            .drop(columns=['ExposurePhecode'])
            .merge(exposure_map, on='Phecode', how='left')
        )

    final_output_path = os.path.join(output_path, f"{trait}_{output_prefix}_phewas_enriched_phecode.csv")
    formatted.to_csv(final_output_path, sep='\t', index=False)
    logging.info(
        'Generated final feature selection file with %d significant phecodes: %s',
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

    logging.info('\n# 2. Resolving the phenotype exposure for Phe-PheWAS')
    phecode_map = load_phecode_map(args.phecode_map_file)
    exposure_phecodes = resolve_exposure_phecodes(phecode_map)

    logging.info('\n# 3. Loading cohort phecodes')
    subset_df = load_binary_phecodes(args.phecode_binary_feather_file, cohort_df['grid'].astype(str).tolist())
    active_exposure_phecodes = resolve_available_exposure_phecodes(
        subset_df=subset_df,
        exposure_phecodes=exposure_phecodes
    )

    subset_df = subset_df[subset_df['grid'].isin(cohort_df['grid'])].copy()
    cohort_df = cohort_df[cohort_df['grid'].isin(subset_df['grid'])].copy()

    logging.info('\n# 4. Writing PheTK input files')
    counts_file_path = write_counts_file(
        subset_df=subset_df,
        output_path=args.output_path,
        output_prefix=args.output_prefix
    )

    logging.info('\n# 5. Running Phe-PheWAS')
    formatted_results = []
    exposure_list = active_exposure_phecodes if active_exposure_phecodes else [None]
    if not active_exposure_phecodes:
        logging.warning(
            'No valid exposure phecodes were configured. Falling back to case/control label for PheWAS.'
        )

    for exposure_phecode in exposure_list:
        exposure_cohort_df, exposure_label, analysis_mode = build_exposure_cohort(
            cohort_df=cohort_df,
            subset_df=subset_df,
            exposure_phecode=exposure_phecode
        )

        if int(exposure_cohort_df['exposure'].sum()) == 0:
            logging.warning('Skipping exposure %s because it has zero exposed subjects.', exposure_label)
            continue

        cohort_file_path = write_cohort_file(
            cohort_df=exposure_cohort_df,
            output_path=args.output_path,
            output_prefix=args.output_prefix,
            exposure_label=exposure_label
        )

        try:
            phewas_output_path = run_phewas(
                cohort_df=exposure_cohort_df,
                cohort_file_path=cohort_file_path,
                counts_file_path=counts_file_path,
                output_path=args.output_path,
                output_prefix=args.output_prefix,
                exposure_label=exposure_label
            )
        except Exception as e:
            logging.error('Error running PheTK PheWAS for exposure %s: %s', exposure_label, e)
            sys.exit(1)

        excluded_for_features = active_exposure_phecodes if analysis_mode == 'phe_phewas' else []
        formatted_results.append(
            format_single_phewas_output(
                results_file=phewas_output_path,
                excluded_phecodes=excluded_for_features,
                exposure_label=exposure_label
            )
        )

    logging.info('\n# 6. Formatting Phe-PheWAS output for ML feature selection')
    combine_feature_selection_outputs(
        formatted_results=formatted_results,
        output_path=args.output_path,
        trait=args.trait,
        output_prefix=args.output_prefix
    )

    time_elapsed = time.time() - start_time
    logging.info('\n# Done. Finished in %.2f minutes', time_elapsed / 60)


if __name__ == '__main__':
    main()
