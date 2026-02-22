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
df = pd.read_csv(path_rozee / 'rozeena_20260212_clean.csv')

## parse conversation
df["messages_parsed"] = df["messages"].apply(ast.literal_eval)
df["n_turns"] = df["messages_parsed"].apply(len)
(df["n_turns"] - df['message_count']).unique()
df.loc[df["n_turns"]>=5,:].shape
df.loc[df["n_turns"]>=10,:].shape

# temporality_check
def analyze_temporality_conversation(messages):
    if not messages or len(messages) < 2:
        return pd.Series({
            "conversation_duration_min": 0,
            "max_gap_min": 0,
            "gap_over_10min": False,
            "gap_over_30min": False,
            "gap_over_60min": False
        })

    # Extract timestamps
    timestamps = []
    for m in messages:
        if "timestamp" in m:
            timestamps.append(datetime.fromisoformat(m["timestamp"]))

    if len(timestamps) < 2:
        return pd.Series({
            "conversation_duration_min": 0,
            "max_gap_min": 0,
            "gap_over_10min": False,
            "gap_over_30min": False,
            "gap_over_60min": False
        })

    # Sort timestamps just in case
    timestamps = sorted(timestamps)

    # 1️⃣ Total duration
    duration_min = (timestamps[-1] - timestamps[0]).total_seconds() / 60

    # 2️⃣ Compute pairwise gaps
    gaps = []
    for i in range(1, len(timestamps)):
        gap = (timestamps[i] - timestamps[i - 1]).total_seconds() / 60
        gaps.append(gap)

    max_gap = max(gaps) if gaps else 0

    return pd.Series({
        "conversation_duration_min": duration_min,
        "max_gap_min": max_gap,
        "gap_over_10min": max_gap > 10,
        "gap_over_30min": max_gap > 30,
        "gap_over_60min": max_gap > 60
    })

# Apply to dataframe
df[[
    "conversation_duration_min",
    "max_gap_min",
    "gap_over_10min",
    "gap_over_30min",
    "gap_over_60min"
]] = df["messages_parsed"].apply(analyze_temporality_conversation)

## push pull
def label_push_pull(msgs):
    if not msgs or len(msgs) == 0:
        return None  # handle empty conversations

    first_turn = msgs[0]

    if "user" in first_turn:
        return "pull"
    elif "rozeena" in first_turn:
        return "push"
    else:
        return None  # in case structure is unexpected


df["push_pull"] = df["messages_parsed"].apply(label_push_pull)
df["push_pull"].value_counts(dropna=False)

# create the push only version
dfp0 = df.loc[df['push_pull']=='push',:]
dfp = dfp0.loc[dfp0['n_turns']>=3,:]
dfu = df.loc[df['push_pull']=='pull',:]


def extract_conversation_turns(msgs, speaker='user'):
    if not msgs:
        return pd.Series([None, None, None, None])

    # Collect Rozeena speaking turns in order
    rozeena_turns = [turn[speaker] for turn in msgs if speaker in turn]

    if not rozeena_turns:
        return pd.Series([None, None, None, None])

    # All Rozeena text merged
    all_rozeena = "||".join(rozeena_turns)

    # 1st, 2nd, 3rd+
    first_turn = rozeena_turns[0] if len(rozeena_turns) >= 1 else None
    second_turn = rozeena_turns[1] if len(rozeena_turns) >= 2 else None
    third_plus = "||".join(rozeena_turns[2:]) if len(rozeena_turns) >= 3 else None

    return pd.Series([all_rozeena, first_turn, second_turn, third_plus])

df[["r_all", "r_t1", "r_t2", "r_t3u"]] = df["messages_parsed"].apply(extract_conversation_turns, speaker='rozeena')
df[["u_all", "u_t1", "u_t2", "u_t3u"]] = df["messages_parsed"].apply(extract_conversation_turns, speaker='user')
dfp[["r_all", "r_t1", "r_t2", "r_t3u"]] = dfp["messages_parsed"].apply(extract_conversation_turns, speaker='rozeena')
dfp[["u_all", "u_t1", "u_t2", "u_t3u"]] = dfp["messages_parsed"].apply(extract_conversation_turns, speaker='user')
dfu[["r_all", "r_t1", "r_t2", "r_t3u"]] = dfu["messages_parsed"].apply(extract_conversation_turns, speaker='rozeena')
dfu[["u_all", "u_t1", "u_t2", "u_t3u"]] = dfu["messages_parsed"].apply(extract_conversation_turns, speaker='user')

## check the length of text
df['r_len'] = df['r_all'].apply(lambda x: len(str(x).split(' ')))
df['r_turns'] = df['r_all'].apply(lambda x: len(str(x).split('||')))
df['u_len'] = df['u_all'].apply(lambda x: len(str(x).split(' ')))
df['u_turns'] = df['u_all'].apply(lambda x: len(str(x).split('||')))


## rule 4 -- redaction first
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(df['r_all'][9])
for ent in doc.ents:
    print(ent.text, ent.label_)

url_pattern = re.compile(r"https?://\S+")
location_pattern = re.compile(r"\b(Pakistan|Karachi|Lahore|Islamabad)\b", re.IGNORECASE)
company_suffix_pattern = re.compile(
    r"\b[A-Z][A-Za-z&,\- ]+(Ltd|Limited|Inc|Corporation|Bank|Solutions|Mill|Pvt|PLC)\b",
    re.IGNORECASE
)


def robust_redact(text):
    if not isinstance(text, str):
        return text

    # 1️⃣ Remove URLs
    text = url_pattern.sub("[URL]", text)

    # 2️⃣ Remove known locations
    text = location_pattern.sub("[LOCATION]", text)

    # 3️⃣ Remove company suffix matches
    text = company_suffix_pattern.sub("[ORG]", text)

    # 4️⃣ spaCy NER pass
    doc = nlp(text)
    redacted = text

    for ent in sorted(doc.ents, key=lambda x: x.start_char, reverse=True):
        if ent.label_ == "PERSON":
            redacted = redacted[:ent.start_char] + "[PERSON]" + redacted[ent.end_char:]
        elif ent.label_ in ["ORG", "GPE", "LOC", "FAC"]:
            redacted = redacted[:ent.start_char] + "[ORG]" + redacted[ent.end_char:]

    return redacted

df["r_all_redacted"] = df["r_all"].apply(robust_redact)

## check data
print('\n'.join(df['r_all'][2].split('||')))
print('\n'.join(robust_redact(df['r_all'][2]).split('||')))
## insight: pattern detection for: *Deputy Manager – Accounts & Finance* at *Aisha Steel Mill Limited Karachi, Pakistan* in Karachi;
# *Deputy Manager – Accounts & Finance* at *Aisha Steel Mill Limited Karachi, Pakistan*, Karachi.

print('\n'.join(df['r_all'][26].split('||')))
print('\n'.join(robust_redact(df['r_all'][2]).split('||')))

df.to_csv(path_rozee / 'data_cleaned.csv', index=False)


### visaulizing basic distributions
# histogram of turns
plt.figure()
plt.hist(df["n_turns"].dropna(), bins=30)
plt.xlabel("Number of Turns")
plt.ylabel("Frequency")
plt.title("Distribution of Conversation Turns")
min_turn = int(df["n_turns"].min())
max_turn = int(df["n_turns"].max())
plt.xticks(np.arange(0, max_turn + 1, 5))
plt.savefig(path_rozee/ "n_turns_histogram.png", dpi=300)

# histogram of conversation duration
plt.figure()
plt.hist(df["conversation_duration_min"].dropna(), bins=30)
plt.xlabel("conversation_duration_min")
plt.ylabel("Frequency")
plt.title("Distribution of Conversation")
plt.savefig(path_rozee/ "conversation_duration_min_histogram.png", dpi=300)

df['gap_over_10min'].value_counts(normalize=True)
df['gap_over_30min'].value_counts(normalize=True)
df['gap_over_60min'].value_counts(normalize=True)
df['gap_over_10min'].value_counts()
df['gap_over_30min'].value_counts()
df['gap_over_60min'].value_counts()

df.groupby(['tone'])['conversation_duration_min'].agg({'mean', 'std', 'median', 'max'})
groups = [group["conversation_duration_min"].values
          for name, group in df.groupby("tone")]
f_stat, p_value = stats.f_oneway(*groups)
print("F-statistic:", f_stat)
print("p-value:", p_value)
posthoc = pg.pairwise_gameshowell(dv='conversation_duration_min',
                                  between='tone',
                                  data=df)
print(posthoc[['A', 'B', 'mean(A)', 'mean(B)', 'pval']])

## print long to echk if the conversations make sense
df.loc[df['n_turns']>=15,:]
df.iloc[26]['u_all']
df.iloc[26]['r_all']
print('\n'.join(df.iloc[26]['messages'].split('{')))
df.iloc[26]['tone']

## create histogram for word length
plt.figure()
plt.hist(df["r_len"].dropna(), bins=30)
plt.xlabel("Number of words, Rozeena")
plt.ylabel("Frequency")
plt.title("Distribution of Number of Words, R")
#min_turn = int(df["r_len"].min())
#max_turn = int(df["r_len"].max())
#plt.xticks(np.arange(0, max_turn + 1, 5))
plt.savefig(path_rozee/ "r_len_histogram.png", dpi=300)
## create histogram
plt.figure()
plt.hist(df["u_len"].dropna(), bins=30)
plt.xlabel("Number of words, Users")
plt.ylabel("Frequency")
plt.title("Distribution of Number of Words, U")
plt.savefig(path_rozee/ "u_len_histogram.png", dpi=300)
## group by tone
df.groupby(['tone'])['r_len'].agg({'mean', 'std', 'median', 'max'})
df.groupby(['tone'])['u_len'].agg({'mean', 'std', 'median', 'max'})


