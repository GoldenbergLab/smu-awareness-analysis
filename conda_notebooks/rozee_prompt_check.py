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

client = OpenAI(
    api_key="xx",
)
path_rozee = Path('D:\\my research\\rozee\\RozeeGPT_Chat\\260212_dashboard_download_pilot1')
df = pd.read_csv(path_rozee / 'data_clean.csv')

# rule 1: check the second follow up
pattern_T1 = re.compile(r"\b(explor\w*|different\w*|common\w*|career\w*|progress\w*)\b", re.IGNORECASE)
pattern_T2 = re.compile(r"\b(feel\w*|felt|unclear\w*|frustrat\w*)\b", re.IGNORECASE)
pattern_T3 = re.compile(r"\b(stay\w*|engag\w*|process\w*|creat\w*|opportunit\w*)\b", re.IGNORECASE)


def check_keyword_match(row, col='r_t2'):
    text = str(row[col]) if pd.notnull(row[col]) else ""
    tone = row["tone"]

    if tone == "T1 - Neutral Informational":
        return bool(pattern_T1.search(text))

    elif tone == "T2 - Empathic Informational":
        return bool(pattern_T2.search(text))

    elif tone == "T3 - Motivational Informational":
        return bool(pattern_T3.search(text))

    else:
        return np.nan
##do it for all
df["keyword_match_r2"] = df.apply(check_keyword_match, axis=1, args=("r_t2",))
df["keyword_match_rall"] = df.apply(check_keyword_match, axis=1, args=("r_all",))
df["keyword_match_r2"].value_counts(dropna=False)
df["keyword_match_rall"].value_counts(dropna=False)
pd.crosstab(df["tone"], df["keyword_match_r2"])

for tone, subset in df[df["keyword_match_r2"] == True].groupby("tone"):
    print(f"\n===== Tone: {tone} =====")

    examples = subset["r_t2"].sample(
        n=min(2, len(subset)),
        random_state=42
    )

    for i, text in enumerate(examples, 1):
        print(f"\n  Example {i}: {text}\n")
## do it for push only
dfp["keyword_match_r2"] = dfp.apply(check_keyword_match, axis=1, args=("r_t2",))
dfp["keyword_match_rall"] = dfp.apply(check_keyword_match, axis=1, args=("r_all",))
dfp["keyword_match_r2"].value_counts(dropna=False)
dfp["keyword_match_rall"].value_counts(dropna=False)
pd.crosstab(dfp["tone"], dfp["keyword_match_r2"])

## check violation
pattern_T1_empathy_violation = re.compile(
    r"\b(i understand|i hear you|that sounds difficult|that must be hard|i('m| am) sorry to hear|i know how you feel|it can feel|it may feel|it’s understandable|you('re| are) not alone|that can be frustrating)\b",
    re.IGNORECASE
)
pattern_T1_motivation_violation = re.compile(
    r"\b(you can do (it|this)|keep going|don'?t worry|stay positive|don'?t give up|keep trying|you('ve| have) got this|stay strong)\b",
    re.IGNORECASE
)
pattern_T1_praise_violation = re.compile(
    r"\b(great job|good luck|well done|excellent work|nice work|great work|proud of you|congratulations)\b",
    re.IGNORECASE
)
pattern_T1_normalization_violation = re.compile(
    r"\b(it('s| is) normal to feel|many people feel|that('s| is) understandable|it happens to many people|others feel this way)\b",
    re.IGNORECASE
)


pattern_motivation_violation = re.compile(
    r"\b(keep going|don't give up|do not give up|you('ve| have) got this|stay strong|you can do (it|this)|stay positive|success is coming|push forward)\b",
    re.IGNORECASE
)
pattern_praise_violation = re.compile(
    r"\b(great job|good luck|well done|proud of you|excellent work|nice work|great work)\b",
    re.IGNORECASE
)
pattern_directive_violation = re.compile(
    r"\b(you should|you must|you need to|make sure to|try to|i recommend|consider (doing|applying|trying))\b",
    re.IGNORECASE
)


pattern_T3_empathy_violation = re.compile(
    r"\b(i understand|i know this is hard|that sounds (hard|frustrating|difficult)|"
    r"it must be (hard|difficult)|i('m| am) sorry to hear|"
    r"i hear you|it can feel|it may feel|that must be hard)\b",
    re.IGNORECASE
)
pattern_T3_normalization_violation = re.compile(
    r"\b(it('s| is) understandable|many people feel|you('re| are) not alone|"
    r"it('s| is) normal to feel|that('s| is) completely understandable)\b",
    re.IGNORECASE
)
pattern_T3_consoling_violation = re.compile(
    r"\b(don'?t worry|it will be okay|everything will work out|"
    r"take your time|no pressure|that('s| is) okay)\b",
    re.IGNORECASE
)
pattern_T3_sympathy_violation = re.compile(
    r"\b(i('m| am) here for you|i care|sending support|"
    r"i('m| am) sorry this happened)\b",
    re.IGNORECASE
)


def check_T1_violations(text):
    if pd.isnull(text):
        return False

    text = str(text)

    return (
            bool(pattern_T1_empathy_violation.search(text)) or
            bool(pattern_T1_motivation_violation.search(text)) or
            bool(pattern_T1_praise_violation.search(text)) or
            bool(pattern_T1_normalization_violation.search(text))
    )
def check_T2_violations(text):
    if pd.isnull(text):
        return False

    text = str(text)

    return (
            bool(pattern_motivation_violation.search(text)) or
            bool(pattern_praise_violation.search(text)) or
            bool(pattern_directive_violation.search(text))
    )


def check_T3_violations(text):
    if pd.isnull(text):
        return False

    text = str(text)

    return (
            bool(pattern_T3_empathy_violation.search(text)) or
            bool(pattern_T3_normalization_violation.search(text)) or
            bool(pattern_T3_consoling_violation.search(text)) or
            bool(pattern_T3_sympathy_violation.search(text))
    )
def check_policy_violation(row, col="r_all"):
    text = row[col]
    tone = row["tone"]

    if pd.isnull(text):
        return False

    text = str(text)

    if tone == "T1 - Neutral Informational":
        return check_T1_violations(text)

    elif tone == "T2 - Empathic Informational":
        return check_T2_violations(text)

    elif tone == "T3 - Motivational Informational":
        return check_T3_violations(text)

    else:
        return False

df["prompt_violation"] = df.apply(
    check_policy_violation,
    axis=1
)
df["prompt_violation"].value_counts()
pd.crosstab(df["tone"], df["prompt_violation"])
pd.crosstab(df["tone"], df["prompt_violation"], normalize="index")

###

## check sentiment
# Initialize analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to compute sentiment
def get_sentiment_score(text):
    if pd.isna(text):
        return None
    return analyzer.polarity_scores(text)['compound']

# Apply to your dataframe
df['sentiment_score'] = df['r_all_redacted'].apply(get_sentiment_score)

# Optional: also keep full sentiment breakdown
def get_full_sentiment(text):
    if pd.isna(text):
        return None
    return analyzer.polarity_scores(text)

df['sentiment_full'] = df['r_all_redacted'].apply(get_full_sentiment)
 ## check results
print('\n'.join(df.loc[df['sentiment_score'] == df['sentiment_score'].min(), 'r_all_redacted'].tolist()[0].split('||')))
print('\n'.join(df.loc[df['sentiment_score'] == df['sentiment_score'].max(), 'r_all_redacted'].tolist()[0].split('||')))
print('\n\n'.join(df.loc[df['sentiment_score'] == df['sentiment_score'].max(), 'r_all'].tolist()[0].split('||')))

df.loc[df['sentiment_score'] == df['sentiment_score'].max(), :].iloc[0]

df.loc[df['r_turns']>=3,:].groupby(['tone'])['sentiment_score'].agg({'mean', 'std', 'median', 'max'})
 ## check stats difference
import scipy.stats as stats
import pingouin as pg

groups = [group["sentiment_score"].values
          for name, group in df.loc[df['r_turns']>=3,:].groupby("tone")]

f_stat, p_value = stats.f_oneway(*groups)

print("F-statistic:", f_stat)
print("p-value:", p_value)
posthoc = pg.pairwise_gameshowell(dv='sentiment_score',
                                  between='tone',
                                  data=df.loc[df['r_turns']>=3,:])
print(posthoc[['A', 'B', 'mean(A)', 'mean(B)', 'pval']])
posthoc.to_csv(path_rozee / 'sentiment_posthoc.csv', index=False)

### LLM judget for construct validity
prompt1 = textwrap.dedent("""
You are evaluating the tone of a chatbot response. The chatbot is a job searching assistant.

There are three possible tone categories:

T1 — Neutral Informational
The message should read as coming from an informational job-search assistant.
The message should provide accurate, clear, and practical job-search information.
The message should use neutral, objective, and factual language.
The message should consist primarily of descriptive or explanatory statements rather than expressive or persuasive language.
The message should not express emotions.
The message should not provide encouragement.
The message should not validate the user emotionally.
The message should not express sympathy or reassurance.
The message should not include praise.
The message should not contain personal opinions.
The message should not acknowledge, mirror, or reference the user’s emotional state.
The message should prioritize informational content over emotional or motivational framing.
The message should include structured or logically organized information.
The message should provide factual guidance about job search practices.
The message should not rely on emotional support or motivational framing.

T2 — Empathic Informational
The message maintains an informational baseline while incorporating empathic framing.
The message reflects an understanding of common job-search experiences (e.g., uncertainty, repeated applications, delays, rejection).
The message acknowledges or mirrors the user’s emotional state when emotions are expressed or implied.
The message uses cognitive empathy by recognizing that certain job-search experiences are common and can be difficult.
The message uses restrained emotional mirroring that aligns with the user’s tone.
The message expresses compassion, care, or respectful acknowledgment of difficulty without becoming overly personal.
The message frames empathy as contextual support for information rather than replacing factual guidance.
The message keeps empathy natural, measured, and context-aware rather than exaggerated.
The message does not provide explicit motivational encouragement (e.g., “keep going,” “you’ve got this”).
The message does not frame calls to action as perseverance, persistence, or success-oriented pushing.
The message does not use strong directive or coaching language.
The message does not include praise.
The message does not use overly therapeutic, consoling, or emotionally intense language.
The message does not over-personalize or imply deep emotional intimacy.
The message does not attach empathic statements to every turn indiscriminately.
If empathy appears, it is most prominent in the opening or selectively reintroduced when the user expresses hesitation, delay, or disappointment.

T3 — Motivational Informational
The message provides accurate, clear, and practical job-search information.
The message maintains an informational baseline while incorporating motivational framing.
The message emphasizes forward momentum, progress, effort, or continued engagement in the job search.
The message frames job-search actions (searching, applying, exploring roles) as productive and meaningful steps toward opportunity.
The message highlights how staying active or persistent increases exposure to opportunities or improves outcomes.
The message uses confident, upbeat, and forward-looking language.
The message includes encouragement that promotes action, confidence, or continued engagement.
The message praises effort, action, or progress rather than emotions.
The message frames setbacks, delays, or waiting periods as part of a larger process of growth or advancement.
The message uses energizing and action-oriented language rather than comforting or consoling language.
The message does not acknowledge or validate the user’s emotional state.
The message does not use empathic phrasing (e.g., “I understand,” “that sounds frustrating”).
The message does not focus on emotional hardship, discouragement, or difficulty.
The message does not use therapeutic, sympathy-based, or emotionally consoling language.
The message does not normalize emotions.

Your task:
1. Read the chatbot message.
2. Decide which of the three tone categories it MOST closely matches.
3. Base your judgment on overall tone and framing, not isolated keywords.
4. If the tone is mixed, select the dominant tone.

Return your answer in JSON format only:

{
  "predicted_tone": "T1" or "T2" or "T3",
  "confidence": number between 0 and 1,
}

Chatbot message:
                          """)


def call_once(text, model="gpt-5-mini", temperature=0):
    response = client.responses.create(
        model=model,
        #temperature=temperature,
        input=[
            #{"role": "system", "content": prompt1},
            {"role": "user", "content": f"{prompt1}:\n{text}"}
        ]
    )

    output_text = response.output[1].content[0].text

    try:
        parsed = json.loads(output_text)
    except:
        parsed = {
            "predicted_tone": None,
            "confidence": None,
            "brief_reason": "Parsing error"
        }

    return parsed


def judge_five_times(text, idx=None):
    results = []
    if idx is not None:
        print(f"Processing row {idx}")

    for _ in range(1):
        result = call_once(text)
        results.append(result)
        #time.sleep(0.1)  # small delay to avoid rate issues

    return results

dfs = df.loc[df['r_turns']>=3, :]
dfsa = dfs.iloc[9:]
dfs["llm_judgments"] = [
    judge_five_times(text, idx=i)
    for i, text in enumerate(dfs["r_all_redacted"])
]
dfs['tome_p'] = dfs["llm_judgments"].apply(lambda x: x[0]['predicted_tone'])
dfs['tome_p'].value_counts()
dfs['tone_t'] = dfs['tone'].apply(lambda x: x.split(' ')[0])
dfs['tone_t'].value_counts()

from sklearn.metrics import classification_report

print(classification_report(
    dfs["tone_t"],
    dfs["tome_p"],
    digits=3
))
