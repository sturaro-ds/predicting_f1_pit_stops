# %% [markdown]
# ## 1. Libraries

# %%
import pandas as pd  # dataframe manipulation
import numpy as np  # numerical operations
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # statistical visualization
import statsmodels.api as sts  # statistical modeling
from skimpy import skim  # quick data summary
from data_profiling import ProfileReport  # automated EDA report
from autogluon.tabular import TabularDataset, TabularPredictor  # AutoML for tabular data
from sklearn.metrics import RocCurveDisplay  # ROC curve visualization

import warnings as ww  # warnings control
ww.filterwarnings('ignore')  # suppress warnings

# %% [markdown]
# ## 2. Datasets
# 
# Observation: The challenge reveals the origin of the dataset, maybe make sense to use the original dataset to train the models.

# %%
original = pd.read_csv('data/original.csv')
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train.drop(columns='id', inplace=True)
test.drop(columns='id', inplace=True)

# %%
# Quicly view

print('==================================================================> ORIGINAL:')
display(original.sample(5))

print('\n==================================================================> TRAIN:')
display(train.sample(5))

print('\n==================================================================> TEST:')
display(test.sample(5))

# %%
# Basic transformation, since these resources are definitely categories, not numbers.

original['PitStop'] = original['PitStop'].astype('category')
original['PitNextLap'] = original['PitNextLap'].astype('category')

train['PitStop'] = train['PitStop'].astype('category')
train['PitNextLap'] = train['PitNextLap'].astype('category')

test['PitStop'] = test['PitStop'].astype('category')

# %% [markdown]
# ## 3. Summary with Skim Library
# 
# Skim is a great library for obtaining quick summaries of datasets, offering a more elegant and intuitive alternative to traditional methods using ".info" files.

# %%
skim(original)

# %% [markdown]
# ### INSIGHTS
# #### After, we will drop nan values, because its irrelevant (6% only)

# %%
skim(train)

# %% [markdown]
# ### INSIGHTS
# #### After, we will investigate the columns: Lap Time, Lap Time Variance, and Cumulative Degradation; this outliers might cause problems in the modeling.

# %%
skim(test)

# %% [markdown]
# ### INSIGHTS
# #### Outliers, similar to the training dataset.

# %%
print('Diference between ORIGINAL and TRAIN: ', set(original.columns.sort_values()) - set(train.columns.sort_values()))
print('Diference between ORIGINAL and TEST: ', set(original.columns.sort_values()) - set(test.columns.sort_values()))

# %% [markdown]
# ### INSIGHTS
# #### After, will create this feature in Train and Test dataset, because in original dataset,  

# %%
features = original.select_dtypes(exclude=['float', 'int', 'category']).columns

for col in features:

    if set(original[col]) - set(train[col]):
        print(f'ORIGINAL vs TRAIN | Column: {col} | Diference in ORIGINAL: {set(original[col]) - set(train[col])}')
    else:
        print(f'ORIGINAL vs TRAIN | Column: {col} |  Result: Same Category')

print(100 * '=','\n')

for col in features:

    if set(original[col]) - set(test[col]):
        print(f'ORIGINAL vs TEST | Column: {col} | Diference in ORIGINAL: {set(original[col]) - set(test[col])}')
    else:
        print(f'ORIGINAL vs TEST | Column: {col} | Result: Same Categorys')

print(100 * '=','\n')

for col in features:

    if set(train[col]) - set(test[col]):
        print(f'TRAIN vs TEST | Column: {col} | Diference in TRAIN: {set(train[col]) - set(test[col])}')
    else:
        print(f'TRAIN vs TEST | Column: {col} | Result: Same Categorys')

# %% [markdown]
# ### INSIGHTS
# #### Pre-season data may exhibit different distributions and cause problems in training, therefore these observations will be discarded.

# %% [markdown]
# ## 4. Data Wrangling, Plots anda Insights

# %%
# Droping nan values

original_dw = original.copy()
original_dw.dropna(inplace=True)

# %%
# Normalized TyreLife Feature

train_dw = train.copy()
test_dw = test.copy()

def normalized_tyrelife(df):

    df['Normalized_TyreLife'] = (
        df.groupby('Stint')['TyreLife']
        .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    )

normalized_tyrelife(train_dw)
normalized_tyrelife(test_dw)

# %%
# Drop pre-seasons data

original_dw = original_dw[
    ~original_dw['Race'].isin([
        'Pre-Season Test',
        'Pre-Season Track Session'
    ])
]

# %%
train_original_dw = pd.concat([train_dw, original_dw], axis=0).reset_index(drop=True)
skim(train_original_dw)    

# %%
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
axes = axes.flatten()

cat_cols = ['Compound', 'Race', 'PitStop']

for i, col in enumerate(cat_cols):
    ct = pd.crosstab(train_original_dw[col], train_original_dw['PitNextLap'], normalize='index')
    ct.plot(kind='barh', ax=axes[i], colormap='Set1')
    axes[i].set_title(col)
    axes[i].set_ylabel('')
    axes[i].legend(title='PitNextLap')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle('Distribution of PitNextLap by Categorical Variables', fontsize=20, y=1.03)

plt.tight_layout()
#plt.savefig('/kaggle/working/Distribution of PitNextLap by Categorical Variables.png', bbox_inches='tight')
#plt.close()
plt.show()


# %% [markdown]
# Concise analysis focused on relevant categorical variables:
# 
# 
# * Compound
#     Clear signal. Softer compounds (e.g., SOFT) show higher pit probability, while harder compounds (e.g., HARD) show lower. Indicates strong relationship between tire type and pit decisions.
# * Race
#     Noticeable variability across races. Some tracks exhibit higher pit frequencies, suggesting track-specific strategy patterns. Useful as a contextual feature, but not dominant alone.
# * PitStop (current lap)
#     Strong and expected signal. Presence of a pit stop is directly associated with PitNextLap, making it a highly predictive feature (though must be used carefully to avoid leakage depending on definition).
# 
# 
# Summary:
# The most relevant categorical driver is Compound, followed by Race as contextual information. PitStop is highly predictive but should be handled cautiously due to its direct relationship with the target.

# %%
corr = train_original_dw.corr(numeric_only=True)

plt.figure(figsize=(16, 10))
ax = sns.heatmap(
    corr,
    cmap="Purples",
    annot=True,
    fmt=".2f"
)

target = 'PitNextLap'
idx = corr.index.get_loc(target)

ax.hlines(idx, *ax.get_xlim(), colors='red', linewidth=3)
ax.hlines(idx+1, *ax.get_xlim(), colors='red', linewidth=3)

ax.vlines(idx, *ax.get_ylim(), colors='red', linewidth=3)
ax.vlines(idx+1, *ax.get_ylim(), colors='red', linewidth=3)

plt.title("Correlation Between Numerical Variables\n", size=20)

#plt.savefig('/kaggle/working/Correlation Between Numerical Variables.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# Key variables related to PitNextLap:
# 
# * TyreLife (0.27) and Normalized_TyreLife (0.21)
#     Strongest signals. Higher values are associated with higher pit probability → core drivers of the model.
# * LapNumber (0.25) and RaceProgress (0.17)
#     Capture race phase. Pit stops are more likely in later stages → important but redundant (keep one, preferably RaceProgress).
# * Stint (0.18)
#     Moderate signal. Reflects progression within tire usage cycles → useful as contextual feature.
# * Cumulative_Degradation (-0.16)
#     Weak but meaningful inverse relationship. May become more useful when transformed (e.g., rate or acceleration).
# 
# Summary:
# The most relevant variables are TyreLife (or its normalized version) and RaceProgress (or LapNumber), supported by Stint. The signal is distributed across these features, indicating that pit decisions depend on a combination of tire wear and race phase, rather than any single variable.

# %%
features_num = train_original_dw.select_dtypes(include="number").columns

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12))
axes = axes.flatten()

for ax, col in zip(axes, features_num):
    sns.kdeplot(
        data=train_original_dw,
        x=col,
        hue="PitNextLap",
        fill=True,
        palette='bwr',
        common_norm=False,
        alpha=0.4,
        ax=ax
    )
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Densidade")

for ax in axes[len(features_num):]:
    ax.set_visible(False)

fig.suptitle("KDE Distribution of Numerical Variables by PitNextLap", fontsize=20, y=1.02)
plt.tight_layout()
#plt.savefig('/kaggle/working/KDE Distribution of Numerical Variables by PitNextLap.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [markdown]
# The KDE plots show that the strongest predictive signals for PitNextLap come from variables related to tire degradation and race progression.
# 
# * The chart shows that 2023 behaves differently from other years, with almost no instances of PitNextLap = 1, indicating a strong class imbalance for that year. This suggests either a data issue (e.g., incomplete or inconsistent labeling) or a distribution shift. As a result, the model may learn a spurious pattern such as “if Year = 2023, then no pit,” which can lead to bias. Overall, Year does not appear to provide meaningful predictive value and may introduce noise or misleading signals. It is safer to remove or carefully handle this feature.
# 
# * TyreLife and Normalized_TyreLife provide clear separation between classes, with higher values strongly associated with pit stops.
# * RaceProgress and LapNumber also show strong discriminatory power, indicating that pit stops are more likely in later stages of the race.
# 
# Some variables present moderate signal:
# 
# * Stint, Position, and Position_Change show slight differences between classes but are not strong predictors on their own.
# 
# Other variables are weak or problematic:
# 
# * LapTime is heavily skewed due to extreme outliers, reducing its usefulness.
# * LapTime_Delta and Cumulative_Degradation are highly concentrated and show limited separation.
# * Year does not exhibit meaningful predictive patterns.
# 
# There is also clear redundancy:
# 
# * TyreLife vs Normalized_TyreLife
# * LapNumber vs RaceProgress
# 
# Only one variable from each pair should be used to avoid multicollinearity.
# 
# Overall, the problem is primarily driven by tire wear dynamics and race phase, suggesting that feature engineering should focus on temporal behavior (lags, rolling statistics, and degradation trends) rather than static variables.

# %% [markdown]
# ## 5. EDA with Profile Report

# %%
# You can save and visualize on html using .to_file(output_file='profile_train_dw.html')
#reporter_train_dw = ProfileReport(df=train_dw).to_notebook_iframe()
#reporter_train_original_dw = ProfileReport(df=train_original_dw).to_notebook_iframe()

# %% [markdown]
# ## 6. Feature Engineering

# %%
def feature_engineering(df):

    df = df.copy()

    # 1) Progress features, both race-level and stint-level
    df['RaceProgress'] = df['LapNumber'] / df.groupby('Race')['LapNumber'].transform('max')
    df['StintProgress'] = df['TyreLife'] / df.groupby(['Driver', 'Race', 'Stint'])['TyreLife'].transform('max')

    # 2) Time since / until pit stop
    df['PitStop_num'] = df['PitStop'].astype(int)

    # Since last pit stop
    df['SinceLastPit'] = df.groupby(['Driver', 'Race'])['PitStop_num'].cumsum()

    # Until next pit stop; label shift, use only for analysis/training if there is no leakage
    df['ToNextPit'] = df.groupby(['Driver', 'Race'])['PitStop_num'].shift(-1)

    # 3) Recent trend features, using lags and rolling windows
    grp = df.groupby(['Driver', 'Race'])

    df['LapTime_lag1'] = grp['LapTime (s)'].shift(1)
    df['LapTime_Delta_lag1'] = grp['LapTime_Delta'].shift(1)

    df['LapTime_roll3'] = grp['LapTime (s)'].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    df['Delta_roll3'] = grp['LapTime_Delta'].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    # 4) Accelerating degradation
    df['Degradation_Accel'] = grp['LapTime_Delta'].diff()

    # 5) Gap to competitors, race context
    df['Pos_Change_Recent'] = grp['Position_Change'].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    )

    # 6) Important interaction features
    df['TyreLife_x_Compound'] = df['TyreLife'] * df['Compound'].astype('category').cat.codes
    df['StintProgress_x_Position'] = df['StintProgress'] * df['Position']

    # Lap time normalized by race average
    df['LapTime_norm_race'] = df['LapTime (s)'] / df.groupby('Race')['LapTime (s)'].transform('mean')

    df['LateRace'] = (df['RaceProgress'] > 0.75).astype(int)
    df['EndOfStint'] = (df['StintProgress'] > 0.8).astype(int)

    top_compound = df['Compound'].value_counts().head(3).index
    df['Compound_grp'] = df['Compound'].where(df['Compound'].isin(top_compound), 'OTHER')

    df.fillna(0, inplace=True)

    return df


# %%
# I will test some aproachings...
train_dw_without_driver = train_dw.drop(columns='Driver')
train_original_dw_without_driver = train_original_dw.drop(columns='Driver')
test_dw_without_driver = test_dw.drop(columns='Driver')

train_dw_fe = train_dw.copy()
train_dw_fe = feature_engineering(train_dw_fe)

train_original_dw_fe = train_original_dw.copy()
train_original_dw_fe = feature_engineering(train_original_dw_fe)

test_dw_fe = test_dw.copy()
test_dw_fe = feature_engineering(test_dw_fe)

# %% [markdown]
# ## 7. AutoML with Autogluon
# 

# %% [markdown]
# ### APROACHING 01

# %%
# APROACHING 01: TRAIN WITH DRIVER AND WITHOUT FEATURE ENGINEERING

train_dw_td = TabularDataset(train_dw.sample(frac=0.5, random_state=42))

races = train_dw_td['Race'].unique()
np.random.seed(42)
val_races = np.random.choice(races, size=int(len(races)*0.2), replace=False)

val_df = train_dw_td[train_dw_td['Race'].isin(val_races)]
train_dw_td = train_dw_td[~train_dw_td['Race'].isin(val_races)]

automl1 = TabularPredictor(
    label='PitNextLap',
    problem_type='binary',
    eval_metric='roc_auc',
    verbosity=2,
    path='autogluon/aproaching-01')

automl1.fit(
    train_data=train_dw_td,
    tuning_data=val_df,
    use_bag_holdout=True,
    presets='best_quality',
    time_limit=3600,
    num_bag_folds=3,
    num_stack_levels=1,
    hyperparameters = {
        'GBM': [{}, {'extra_trees': True}],
        'CAT': {},
        'XGB': {},
        'XT': {}
    })

# %%
# --- data ---
leaderboard_automl1 = automl1.leaderboard(silent=True)
fi_automl1 = automl1.feature_importance(train_dw_td)

y_true = val_df['PitNextLap']
y_proba = automl1.predict_proba(val_df)

# ensure we select the positive class probability
if hasattr(y_proba, "shape") and len(y_proba.shape) > 1:
    y_proba = y_proba.iloc[:, 1]

# ensure we select the positive class probability
if hasattr(y_proba, "shape") and len(y_proba.shape) > 1:
    y_proba = y_proba.iloc[:, 1]

# style
sns.set(style="whitegrid")

# --- subplots ---
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# color palettes
palette_models = sns.color_palette("Purples_r", len(leaderboard_automl1))
palette_fi = sns.color_palette("mako", len(fi_automl1))

# 1) Leaderboard
sns.barplot(
    data=leaderboard_automl1,
    x='score_val',
    y='model',
    palette=palette_models,
    ax=axes[0]
)

axes[0].set_title('LEADERBOARD BY ROC AUC SCORE', fontweight='bold')
axes[0].set_xlabel('ROC AUC')

# add value label to each bar
for i, v in enumerate(leaderboard_automl1['score_val']):
    axes[0].text(
        v + 0.002,
        i,
        f"{v:.3f}",
        va='center',
        fontsize=9
    )

# 2) Feature Importance
sns.barplot(
    x='importance',
    y=fi_automl1.index,
    data=fi_automl1,
    palette=palette_fi,
    ax=axes[1]
)

axes[1].set_title('FEATURE IMPORTANCE', fontweight='bold')

# 3) ROC Curve
RocCurveDisplay.from_predictions(
    y_true,
    y_proba,
    ax=axes[2],
    color='#6100b9'  # purple tone
)

axes[2].plot([0, 1], [0, 1], linestyle='--', color='gray')
axes[2].set_title('ROC CURVE', fontweight='bold')

plt.tight_layout()
#plt.savefig('/kaggle/working/Plots Aproaching 01.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Submission Aproaching 01

sub_aproaching01 = automl1.predict_proba(test_dw)[1]
sub_id = pd.read_csv('data/test.csv')
sub_aproaching01 = pd.DataFrame(
    {
        'id': sub_id['id'], 
        'PitNextLap': sub_aproaching01
        }
)
sub_aproaching01.to_csv('submission/submission-ap01.csv', index=False)

# %% [markdown]
# ### APROACHING 02

# %%
# APROACHING 02: TRAIN WITHOUT DRIVER AND WITHOUT FEATURE ENGINEERING

train_dw_without_driver_td = TabularDataset(train_dw_without_driver.sample(frac=0.6, random_state=42))

races = train_dw_without_driver_td['Race'].unique()
np.random.seed(42)
val_races = np.random.choice(races, size=int(len(races)*0.2), replace=False)

val_df = train_dw_without_driver_td[train_dw_without_driver_td['Race'].isin(val_races)]
train_dw_without_driver_td = train_dw_without_driver_td[~train_dw_without_driver_td['Race'].isin(val_races)]

automl2 = TabularPredictor(
    label='PitNextLap',
    problem_type='binary',
    eval_metric='roc_auc',
    verbosity=1,
    path='autogluon/aproaching-02')

automl2.fit(
    train_data=train_dw_without_driver_td,
    tuning_data=val_df,
    use_bag_holdout=True,
    presets='best_quality',
    time_limit=3600,
    num_bag_folds=3,
    num_stack_levels=1,
    hyperparameters = {
        'GBM': [{}, {'extra_trees': True}],
        'CAT': {},
        'XGB': {},
        'XT': {}
    })

# %%
# --- data ---
leaderboard_automl2 = automl2.leaderboard(silent=True)
fi_automl2 = automl2.feature_importance(train_dw_without_driver_td)

y_true = val_df['PitNextLap']
y_proba = automl2.predict_proba(val_df)

# ensure we select the positive class probability
if hasattr(y_proba, "shape") and len(y_proba.shape) > 1:
    y_proba = y_proba.iloc[:, 1]

# ensure we select the positive class probability
if hasattr(y_proba, "shape") and len(y_proba.shape) > 1:
    y_proba = y_proba.iloc[:, 1]

# style
sns.set(style="whitegrid")

# --- subplots ---
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# color palettes
palette_models = sns.color_palette("Purples_r", len(leaderboard_automl2))
palette_fi = sns.color_palette("mako", len(fi_automl2))

# 1) Leaderboard
sns.barplot(
    data=leaderboard_automl2,
    x='score_val',
    y='model',
    palette=palette_models,
    ax=axes[0]
)

axes[0].set_title('LEADERBOARD BY ROC AUC SCORE', fontweight='bold')
axes[0].set_xlabel('ROC AUC')

# add value label to each bar
for i, v in enumerate(leaderboard_automl2['score_val']):
    axes[0].text(
        v + 0.002,
        i,
        f"{v:.3f}",
        va='center',
        fontsize=9
    )

# 2) Feature Importance
sns.barplot(
    x='importance',
    y=fi_automl2.index,
    data=fi_automl2,
    palette=palette_fi,
    ax=axes[1]
)

axes[1].set_title('FEATURE IMPORTANCE', fontweight='bold')

# 3) ROC Curve
RocCurveDisplay.from_predictions(
    y_true,
    y_proba,
    ax=axes[2],
    color='#6100b9'  # purple tone
)

axes[2].plot([0, 1], [0, 1], linestyle='--', color='gray')
axes[2].set_title('ROC CURVE', fontweight='bold')

plt.tight_layout()
#plt.savefig('/kaggle/working/Plots Aproaching 02.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Submission Aproaching 02

sub_aproaching02 = automl2.predict_proba(test_dw)[1]
sub_aproaching02 = pd.DataFrame(
    {
        'id': sub_id['id'], 
        'PitNextLap': sub_aproaching02
        }
)
sub_aproaching02.to_csv('submission/submission-ap02.csv', index=False)

# %% [markdown]
# ### APROACHING 03

# %%
# APROACHING 03: TRAIN WITH FEATURE ENGINEERING

train_dw_fe_td = TabularDataset(train_dw_fe.sample(frac=0.7, random_state=42))

races = train_dw_fe_td['Race'].unique()
np.random.seed(42)
val_races = np.random.choice(races, size=int(len(races)*0.2), replace=False)

val_df = train_dw_fe_td[train_dw_fe_td['Race'].isin(val_races)]
train_dw_fe_td = train_dw_fe_td[~train_dw_fe_td['Race'].isin(val_races)]

automl3 = TabularPredictor(
    label='PitNextLap',
    problem_type='binary',
    eval_metric='roc_auc',
    verbosity=1,
    path='autogluon/aproaching-03')

automl3.fit(
    train_data=train_dw_fe_td,
    tuning_data=val_df,
    use_bag_holdout=True,
    presets='best_quality',
    time_limit=3600,
    num_bag_folds=3,
    num_stack_levels=1,
    hyperparameters = {
        'GBM': [{}, {'extra_trees': True}],
        'CAT': {},
        'XGB': {},
        'XT': {}
    })

# %%
# --- data ---
leaderboard_automl3 = automl3.leaderboard(silent=True)
fi_automl3 = automl3.feature_importance(train_dw_fe_td)

y_true = val_df['PitNextLap']
y_proba = automl3.predict_proba(val_df)

# ensure we select the positive class probability
if hasattr(y_proba, "shape") and len(y_proba.shape) > 1:
    y_proba = y_proba.iloc[:, 1]

# ensure we select the positive class probability
if hasattr(y_proba, "shape") and len(y_proba.shape) > 1:
    y_proba = y_proba.iloc[:, 1]

# style
sns.set(style="whitegrid")

# --- subplots ---
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# color palettes
palette_models = sns.color_palette("Purples_r", len(leaderboard_automl3))
palette_fi = sns.color_palette("mako", len(fi_automl3))

# 1) Leaderboard
sns.barplot(
    data=leaderboard_automl3,
    x='score_val',
    y='model',
    palette=palette_models,
    ax=axes[0]
)

axes[0].set_title('LEADERBOARD BY ROC AUC SCORE', fontweight='bold')
axes[0].set_xlabel('ROC AUC')

# add value label to each bar
for i, v in enumerate(leaderboard_automl3['score_val']):
    axes[0].text(
        v + 0.002,
        i,
        f"{v:.3f}",
        va='center',
        fontsize=9
    )

# 2) Feature Importance
sns.barplot(
    x='importance',
    y=fi_automl3.index,
    data=fi_automl3,
    palette=palette_fi,
    ax=axes[1]
)

axes[1].set_title('FEATURE IMPORTANCE', fontweight='bold')

# 3) ROC Curve
RocCurveDisplay.from_predictions(
    y_true,
    y_proba,
    ax=axes[2],
    color='#6100b9'  # purple tone
)

axes[2].plot([0, 1], [0, 1], linestyle='--', color='gray')
axes[2].set_title('ROC CURVE', fontweight='bold')

plt.tight_layout()
#plt.savefig('/kaggle/working/Plots Aproaching 03.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Submission Aproaching 03

sub_aproaching03 = automl1.predict_proba(test_dw_fe)[1]
sub_aproaching03 = pd.DataFrame(
    {
        'id': sub_id['id'], 
        'PitNextLap': sub_aproaching03
        }
)
sub_aproaching03.to_csv('submission/submission-ap03.csv', index=False)

# %% [markdown]
# ### APROACHING 04

# %%
# APROACHING 04: TRAIN + ORIGINAL WITHOUT DRIVER AND FEATURE ENGINEERING

train_original_dw_without_driver_td = TabularDataset(train_original_dw_without_driver.sample(frac=0.7, random_state=42))

races = train_original_dw_without_driver_td['Race'].unique()
np.random.seed(42)
val_races = np.random.choice(races, size=int(len(races)*0.2), replace=False)

val_df = train_original_dw_without_driver_td[train_original_dw_without_driver_td['Race'].isin(val_races)]
train_original_dw_without_driver_td = train_original_dw_without_driver_td[~train_original_dw_without_driver_td['Race'].isin(val_races)]

automl4 = TabularPredictor(
    label='PitNextLap',
    problem_type='binary',
    eval_metric='roc_auc',
    verbosity=1,
    path='autogluon/aproaching-04')

automl4.fit(
    train_data=train_original_dw_without_driver_td,
    tuning_data=val_df,
    use_bag_holdout=True,
    presets='best_quality',
    time_limit=3600,
    num_bag_folds=3,
    num_stack_levels=1,
    hyperparameters = {
        'GBM': [{}, {'extra_trees': True}],
        'CAT': {},
        'XGB': {},
        'XT': {}
    })

# %%
# --- data ---
leaderboard_automl4 = automl3.leaderboard(silent=True)
fi_automl4 = automl3.feature_importance(train_dw_fe_td)

y_true = val_df['PitNextLap']
y_proba = automl3.predict_proba(val_df)

# ensure we select the positive class probability
if hasattr(y_proba, "shape") and len(y_proba.shape) > 1:
    y_proba = y_proba.iloc[:, 1]

# ensure we select the positive class probability
if hasattr(y_proba, "shape") and len(y_proba.shape) > 1:
    y_proba = y_proba.iloc[:, 1]

# style
sns.set(style="whitegrid")

# --- subplots ---
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# color palettes
palette_models = sns.color_palette("Purples_r", len(leaderboard_automl4))
palette_fi = sns.color_palette("mako", len(fi_automl4))

# 1) Leaderboard
sns.barplot(
    data=leaderboard_automl4,
    x='score_val',
    y='model',
    palette=palette_models,
    ax=axes[0]
)

axes[0].set_title('LEADERBOARD BY ROC AUC SCORE', fontweight='bold')
axes[0].set_xlabel('ROC AUC')

# add value label to each bar
for i, v in enumerate(leaderboard_automl4['score_val']):
    axes[0].text(
        v + 0.002,
        i,
        f"{v:.3f}",
        va='center',
        fontsize=9
    )

# 2) Feature Importance
sns.barplot(
    x='importance',
    y=fi_automl4.index,
    data=fi_automl4,
    palette=palette_fi,
    ax=axes[1]
)

axes[1].set_title('FEATURE IMPORTANCE', fontweight='bold')

# 3) ROC Curve
RocCurveDisplay.from_predictions(
    y_true,
    y_proba,
    ax=axes[2],
    color='#6100b9'  # purple tone
)

axes[2].plot([0, 1], [0, 1], linestyle='--', color='gray')
axes[2].set_title('ROC CURVE', fontweight='bold')

plt.tight_layout()
#plt.savefig('/kaggle/working/Plots Aproaching 04.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Submission Aproaching 04

sub_aproaching04 = automl3.predict_proba(test_dw_without_driver)[1]
sub_aproaching04 = pd.DataFrame(
    {
        'id': sub_id['id'], 
        'PitNextLap': sub_aproaching04
        }
)
sub_aproaching04.to_csv('submission/submission-ap04.csv', index=False)


