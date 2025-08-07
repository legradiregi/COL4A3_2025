from pathlib import Path

import sys 
sys.path.append(Path().parent.resolve() /'.venv/lib/python3.10/site-packages/')

import polars as pl
import polars.selectors as cs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Literal
from scipy.stats import fisher_exact, kruskal, mannwhitneyu, chi2_contingency


HOME = Path().parent.resolve()
INPUTS = HOME / 'inputs'
OUTPUTS = HOME / 'outputs'
FIGURES = HOME / 'plots'

OUTPUTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

def form(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(
        pl.col('idx').cast(pl.String),
        ).unpivot(
            cs.integer(),
            index=['idx'],
            variable_name='genotype',
            value_name='value'
        ).drop_nulls(
            ).drop(
                'idx'
                )
            
def plot(
    data: pl.DataFrame,
    y_label: str,
    title: str,
    figsize: tuple = (16, 8),
    hue: str = None,
    hue_order: list = None,
    color: str = None,
    custom_palette: list = None,
    ) -> None:

    max_value = data.get_column('value').max()
    upper_limit = (max_value // 20 + 1) * 20
    min_value = data.get_column('value').min()
    
    plt.figure(figsize=figsize)
    sns.set_theme(style="whitegrid")
    g = sns.boxplot(
        data=data,
        x='genotype',
        y='value',
        hue=hue,
        hue_order=hue_order,
        color=color,
        palette=custom_palette,
        legend=False,
        boxprops=dict(alpha=0.3),
        showfliers=False
    )

    sns.stripplot(
        data=data,
        x='genotype',
        y='value',
        hue=hue,
        dodge=False,
        color=color,
        legend=False,
        palette=custom_palette,
        hue_order=hue_order,
        alpha=0.7,
        s=5,

    )
    # g.legend(title=None)
    g.set_ylim(0 if min_value > 0 else min_value-1, upper_limit)   
    g.set_yticks(range(0, upper_limit + 1, 20))
    g.set_yticks(range(0, upper_limit + 1, 5), minor=True)
     
    g.tick_params(axis='x', rotation=45, labelsize=12)
    g.set(xlabel ="", ylabel = y_label)
    plt.xlabel('', fontsize=16)
    plt.ylabel(y_label, fontsize=20)

    g.tick_params(axis='both', labelsize=16)
    # plt.title(title, fontdict={'fontsize': 24, 'fontweight': 'bold'})
    plt.savefig(FIGURES / f'{title}.jpg', 
                dpi=300, 
                bbox_inches = "tight")
    plt.show()

def run_statistics(
    data: pl.DataFrame,
    group_by: str,
    title: str
    ) -> None:
    
    print(f"Statistics for {title}")
    
    results = []
    
    groups = data.get_column(group_by).unique().to_list()

    for group1 in groups:
        for group2 in groups:
            mw = (mannwhitneyu(
                data.filter(pl.col(group_by) == group1).get_column('value').to_list(),
                data.filter(pl.col(group_by) == group2).get_column('value').to_list(),
                method='exact'
            ))
            results.append({
                'reference': group1,
                'compare': group2,
                'statistic': mw.statistic,
                'pvalue': mw.pvalue
            })
            
            res = kruskal(
                data.filter(pl.col(group_by) == group1).get_column('value').to_list(),
                data.filter(pl.col(group_by) == group2).get_column('value').to_list(),
            )
            print(f"Kruskal-Wallis test: H={res.statistic:.4f}, p-value={res.pvalue:.4e}")
            print()
            
            results.append({
                'reference': group1,
                'compare': group2,
                'kruskal_statistic': res.statistic,
                'kruskal_pvalue': res.pvalue
            })
            
    result_df = pl.from_dicts(results)
    result_df.write_csv(OUTPUTS / f'{title}_statistics.csv')


def run_stat_test(
        input_table: np.ndarray,
        test: Literal['chi2', 'fisher']
):

    # Rows = samples, columns = features
    if test == 'chi2':
        chi2, p, dof, expected = chi2_contingency(input_table)

        print(f"Chi² statistic = {chi2:.4f}")
        print(f"p-value = {p:.4e}")
        print(f"Degrees of freedom = {dof}")
        print()
    elif test == 'fisher':

        odds_ratio, p_value = fisher_exact(input_table)

        print(f"Fisher's Exact Test:")
        print(f"Odds ratio = {odds_ratio:.4f}")
        print(f"p-value = {p_value:.4e}")

    else: 
        raise Exception('Invalid test type (chi2/fisher)')