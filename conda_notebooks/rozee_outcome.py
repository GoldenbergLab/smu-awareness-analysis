import textwrap
from pathlib import Path
import pandas as pd
import numpy as np
import ast
import re
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import scipy.stats as stats
import pingouin as pg
from openai import OpenAI
import textwrap
import json
import time

path_rozee = Path('D:\\my research\\rozee\\RozeeGPT_Chat\\260212_dashboard_download_pilot1')
df = pd.read_csv(path_rozee / 'data_cleaned_feb19_yl.csv')
### check the result of outcomes
df['outcome_b'] = 0
df.loc[df['outcome']=='interested', 'outcome_b'] = 1


# Create contingency table
contingency_table = pd.crosstab(df['tone'], df['outcome_b'])
print("Contingency Table:")
print(contingency_table)
# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print("\nChi-square statistic:", chi2)
print("Degrees of freedom:", dof)
print("p-value:", p)
prop_table = df.groupby('tone')['outcome_b'].agg(['mean', 'std', 'count'])
prop_table['se'] = prop_table['std'] / (prop_table['count'] ** 0.5)
print(prop_table)
## plot
tones = prop_table.index
means = prop_table['mean']
sds = prop_table['std']
# Plot
plt.figure(figsize=[10,6])
plt.bar(tones, means)
plt.errorbar(tones, means, yerr=sds, fmt='none', capsize=5)
plt.xlabel("Tone")
plt.ylabel("Proportion Interested (outcome_b=1)")
plt.title("Proportion of Outcome by Tone")
plt.savefig(path_rozee/ "outcome_b_histogram.png", dpi=300)

## send link or not
df['link_b'] = 0
df.loc[df['apply_link_sent']=='Yes', 'link_b'] = 1

# Create contingency table
contingency_table = pd.crosstab(df['tone'], df['link_b'])
print("Contingency Table:")
print(contingency_table)
# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print("\nChi-square statistic:", chi2)
print("Degrees of freedom:", dof)
print("p-value:", p)
prop_table = df.groupby('tone')['link_b'].agg(['mean', 'std', 'count'])
prop_table['se'] = prop_table['std'] / (prop_table['count'] ** 0.5)
print(prop_table)
## plot
tones = prop_table.index
means = prop_table['mean']
sds = prop_table['std']
# Plot
plt.figure(figsize=[10,6])
plt.bar(tones, means)
plt.errorbar(tones, means, yerr=sds, fmt='none', capsize=5)
plt.xlabel("Tone")
plt.ylabel("Linked sent (link_b=1)")
plt.title("Proportion of Link Sent by Tone")
plt.savefig(path_rozee/ "link_b_histogram.png", dpi=300)
