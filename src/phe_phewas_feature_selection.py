import argparse
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl
import yaml
from joblib import Parallel, delayed
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

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
    parser.add_argument('--n_jobs', type=int, default=config.get('phewas_n_jobs', -1),
                        help='Number of parallel jobs for per-phecode regressions')
    parser.add_argument('--max_iter', type=int, default=config.get('phewas_max_iter', 200),
                        help='Maximum iterations for sklearn logistic regression')

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
    """Load the phecode map used for descriptions and excluded ICD resolution."""
    if not phecode_map_file:
        return pd.DataFrame(columns=['ICD', 'Phecode', 'PhecodeString'])

    phecode_map = pd.read_csv(phecode_map_file, dtype={'ICD': str, 'Phecode': str})
    return phecode_map


def resolve_excluded_phecodes(phecode_map):
    """Resolve phenotype-defining phecodes that should be excluded from ML features."""
    excluded_codes = config.get('excluded_codes', {}) or {}
    excluded_phecodes = [str(code).strip() for code in (excluded_codes.get('Phecode', []) or []) if str(code).strip()]

    excluded_icd = excluded_codes.get('ICD', []) or []
    if excluded_icd and not phecode_map.empty and {'ICD', 'Phecode'}.issubset(phecode_map.columns):
        icd_map = phecode_map[['ICD', 'Phecode']].dropna().copy()
        icd_map['ICD'] = icd_map['ICD'].astype(str).str.strip()
        icd_map['Phecode'] = icd_map['Phecode'].astype(str).str.strip()
        matched = icd_map[icd_map['ICD'].isin([str(code).strip() for code in excluded_icd])]
        excluded_phecodes.extend(matched['Phecode'].tolist())

    deduped = []
    for code in excluded_phecodes:
        if code and code not in deduped:
            deduped.append(code)
    return deduped


def build_phecode_description_map(phecode_map):
    """Map phecodes to descriptions for output tables."""
    if phecode_map.empty or 'PhecodeString' not in phecode_map.columns:
        return {}

    desc_df = phecode_map[['Phecode', 'PhecodeString']].dropna().copy()
    desc_df['Phecode'] = desc_df['Phecode'].astype(str).str.strip()
    desc_df = desc_df.drop_duplicates(subset=['Phecode'])
    return dict(zip(desc_df['Phecode'], desc_df['PhecodeString']))


def load_demographics(demographics_file):
    """Load the sex field needed by the regression models."""
    logging.info(f'Loading demographics from {demographics_file}')
    if str(demographics_file).endswith('.csv.gz') or str(demographics_file).endswith('.csv'):
        return pd.read_csv(demographics_file, usecols=['grid', 'gender_source_value'])
    return pd.read_csv(demographics_file, sep='\t', usecols=['grid', 'gender_source_value'])


def build_cohort(cases, controls, demographics_file):
    """
    Build the matched cohort used for Phe-PheWAS.

    The exposure phenotype is the matched case/control label itself.
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


def load_analysis_matrix(feather_file, cohort_df):
    """Load only the matched cohort rows and join covariates using Polars."""
    logging.info(f"Loading binary phecodes from {feather_file} (lazy)")
    cohort_pl = pl.from_pandas(cohort_df[['grid', 'label', 'sex']])
    phecode_pl = (
        pl.scan_ipc(feather_file)
        .with_columns(pl.col('grid').cast(pl.Utf8))
        .join(cohort_pl.lazy(), on='grid', how='inner')
        .collect()
    )
    logging.info('Collected phecode matrix with shape %s', phecode_pl.shape)
    return phecode_pl


def build_candidate_table(phecode_pl, excluded_phecodes, min_case_count, min_total_count):
    """Compute per-phecode counts and prefilter outcomes before regression."""
    phecode_cols = [col for col in phecode_pl.columns if col not in {'grid', 'label', 'sex'}]
    case_mask = pl.col('label') == 1

    agg_exprs = []
    for phecode in phecode_cols:
        agg_exprs.extend([
            pl.col(phecode).sum().alias(f'{phecode}__total'),
            pl.col(phecode).filter(case_mask).sum().alias(f'{phecode}__cases'),
        ])

    counts_row = phecode_pl.select(agg_exprs).to_dicts()[0]
    records = []
    cohort_cases = int(phecode_pl['label'].sum())
    cohort_controls = phecode_pl.height - cohort_cases
    for phecode in phecode_cols:
        total = int(counts_row[f'{phecode}__total'])
        case_count = int(counts_row[f'{phecode}__cases'])
        control_count = total - case_count
        if phecode in excluded_phecodes:
            continue
        if total < min_total_count:
            continue
        if case_count < min_case_count:
            continue
        if control_count < 1:
            continue
        if total >= phecode_pl.height:
            continue
        if case_count >= cohort_cases and control_count >= cohort_controls:
            continue
        records.append({
            'phecode': phecode,
            'cases': case_count,
            'controls': control_count,
            'total': total,
        })

    candidate_df = pd.DataFrame(records)
    if not candidate_df.empty:
        candidate_df.sort_values(by=['cases', 'total'], ascending=[False, False], inplace=True)
    logging.info(
        'Prepared %d candidate phecodes for regression (from %d total columns)',
        len(candidate_df),
        len(phecode_cols)
    )
    return candidate_df, phecode_cols


def fit_single_phecode(y, x, x_design, phecode, case_count, control_count, max_iter):
    """Fit one logistic model and compute a Wald p-value for the exposure coefficient."""
    if np.unique(y).size < 2:
        return None

    try:
        model = LogisticRegression(
            penalty=None,
            solver='newton-cholesky',
            fit_intercept=True,
            max_iter=max_iter
        )
        model.fit(x, y)
        beta = np.concatenate((model.intercept_, model.coef_[0]))
        linear = x_design @ beta
        prob = expit(linear)
        weights = prob * (1.0 - prob)
        fisher = x_design.T @ (weights[:, None] * x_design)
        cov = np.linalg.pinv(fisher)
        se = float(np.sqrt(max(cov[1, 1], 0.0)))
        if not np.isfinite(se) or se == 0:
            return None

        exposure_beta = float(beta[1])
        z_score = exposure_beta / se
        p_value = float(2 * norm.sf(abs(z_score)))
        return {
            'phecode': phecode,
            'beta': exposure_beta,
            'odds_ratio': float(np.exp(exposure_beta)),
            'standard_error': se,
            'z_score': float(z_score),
            'p_value': p_value,
            'cases': int(case_count),
            'controls': int(control_count),
            'count': int(case_count + control_count),
        }
    except Exception:
        return None


def add_multiple_testing_columns(results_df, alpha):
    """Add Bonferroni and Benjamini-Hochberg adjusted p-values."""
    if results_df.empty:
        results_df = results_df.copy()
        results_df['bonferroni_p_value'] = pd.Series(dtype=float)
        results_df['fdr_bh_q_value'] = pd.Series(dtype=float)
        results_df['bonferroni_significant'] = pd.Series(dtype=bool)
        results_df['fdr_bh_significant'] = pd.Series(dtype=bool)
        return results_df

    adjusted = results_df.copy()
    pvals = adjusted['p_value'].to_numpy(dtype=float)
    m = pvals.size

    bonferroni = np.minimum(pvals * m, 1.0)

    order = np.argsort(pvals)
    ranked = pvals[order]
    ranks = np.arange(1, m + 1, dtype=float)
    bh = ranked * m / ranks
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0.0, 1.0)
    fdr = np.empty_like(bh)
    fdr[order] = bh

    adjusted['bonferroni_p_value'] = bonferroni
    adjusted['fdr_bh_q_value'] = fdr
    adjusted['bonferroni_significant'] = adjusted['bonferroni_p_value'] < alpha
    adjusted['fdr_bh_significant'] = adjusted['fdr_bh_q_value'] < alpha
    return adjusted


def run_phewas(matrix_df, candidate_df, output_path, output_prefix, n_jobs, max_iter):
    """Run per-phecode logistic regressions with shared design matrices."""
    phewas_output_path = os.path.join(output_path, f"{output_prefix}_phewas_results.tsv")
    if candidate_df.empty:
        empty_df = pd.DataFrame(
            columns=[
                'phecode', 'beta', 'odds_ratio', 'standard_error', 'z_score', 'p_value',
                'cases', 'controls', 'count', 'bonferroni_p_value', 'fdr_bh_q_value',
                'bonferroni_significant', 'fdr_bh_significant'
            ]
        )
        empty_df.to_csv(phewas_output_path, sep='\t', index=False)
        logging.warning('No phecodes passed the regression prefilter; wrote an empty result file.')
        return phewas_output_path

    x = matrix_df[['label', 'sex']].to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(matrix_df.shape[0], dtype=float), x])

    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(fit_single_phecode)(
            y=matrix_df[phecode].to_numpy(dtype=int),
            x=x,
            x_design=x_design,
            phecode=phecode,
            case_count=case_count,
            control_count=control_count,
            max_iter=max_iter
        )
        for phecode, case_count, control_count in candidate_df[['phecode', 'cases', 'controls']].itertuples(index=False, name=None)
    )
    results = [row for row in results if row is not None]
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.sort_values(by='p_value', ascending=True, inplace=True)
        results_df = add_multiple_testing_columns(
            results_df=results_df,
            alpha=float(config.get('phewas_alpha', 0.05))
        )
    results_df.to_csv(phewas_output_path, sep='\t', index=False)
    logging.info('Phe-PheWAS completed. Results saved to %s', phewas_output_path)
    return phewas_output_path


def format_feature_selection_output(results_file, output_path, trait, output_prefix, phecode_desc_map):
    """Format regression output into the ML feature-selection table."""
    results_df = pd.read_csv(results_file, sep='\t', dtype={'phecode': str})
    if results_df.empty:
        formatted = pd.DataFrame(columns=['Phecode', 'Description', 'Count', 'p.value'])
    else:
        sig_cutoff = config.get('phewas_sig_cutoff', 1e-5)
        formatted = results_df[results_df['p_value'] < sig_cutoff].copy()
        if formatted.empty:
            logging.warning(
                'No phecodes met the significance cutoff. Falling back to all tested phecodes ranked by p-value.'
            )
            formatted = results_df.copy()

        formatted['Description'] = formatted['phecode'].map(phecode_desc_map).fillna('NA')
        formatted.rename(columns={'phecode': 'Phecode', 'cases': 'Count', 'p_value': 'p.value'}, inplace=True)
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

    logging.info('\n# 2. Resolving metadata and leakage exclusions')
    phecode_map = load_phecode_map(args.phecode_map_file)
    excluded_phecodes = resolve_excluded_phecodes(phecode_map)
    phecode_desc_map = build_phecode_description_map(phecode_map)

    logging.info('\n# 3. Loading analysis matrix with Polars')
    phecode_pl = load_analysis_matrix(args.phecode_binary_feather_file, cohort_df)
    matrix_df = phecode_pl.to_pandas()
    matrix_df['grid'] = matrix_df['grid'].astype(str)

    logging.info('\n# 4. Prefiltering phecodes before regression')
    min_case_count = max(1, round(int(matrix_df['label'].sum()) * config.get('min_phecode_frequency', 0.02)))
    min_total_count = max(1, int(config.get('phewas_min_total_count', 2)))
    candidate_df, _ = build_candidate_table(
        phecode_pl=phecode_pl,
        excluded_phecodes=excluded_phecodes,
        min_case_count=min_case_count,
        min_total_count=min_total_count
    )

    logging.info('\n# 5. Running Polars + sklearn Phe-PheWAS')
    results_file = run_phewas(
        matrix_df=matrix_df,
        candidate_df=candidate_df,
        output_path=args.output_path,
        output_prefix=args.output_prefix,
        n_jobs=args.n_jobs,
        max_iter=args.max_iter
    )

    logging.info('\n# 6. Formatting output for ML feature selection')
    format_feature_selection_output(
        results_file=results_file,
        output_path=args.output_path,
        trait=args.trait,
        output_prefix=args.output_prefix,
        phecode_desc_map=phecode_desc_map
    )

    time_elapsed = time.time() - start_time
    logging.info('\n# Done. Finished in %.2f minutes', time_elapsed / 60)


if __name__ == '__main__':
    main()
