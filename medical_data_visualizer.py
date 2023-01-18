import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv', header=0)
# convert height in cm to height in m
# Add 'overweight' column
df['overweight'] = df['weight'] / ((df['height'] / 100) ** 2)
# Modify the column, overweight > 25 then 1 else 0
df.loc[df['overweight'] <= 25, 'overweight'] = 0
df.loc[df['overweight'] > 25, 'overweight'] = 1

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
# modify the column, cholesterol = 1 then 0 else 1
df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1
# modify the column, gluc = 1 then 0 else 1
df.loc[df['gluc'] == 1, 'gluc'] = 0
df.loc[df['gluc'] > 1, 'gluc'] = 1


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    cardio = pd.melt(df.loc[(df['cardio'] == 1)],
                     id_vars=['id'],
                     value_vars=[
                         'active', 'alco', 'cholesterol', 'gluc', 'overweight',
                         'smoke'
                     ])

    no_cardio = pd.melt(df.loc[(df['cardio'] == 0)],
                        id_vars=['id'],
                        value_vars=[
                            'active', 'alco', 'cholesterol', 'gluc', 'overweight',
                            'smoke'
                        ])

    # # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    # df_cat = None

    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('cardio = 0')
    sns.countplot(data=no_cardio, x='variable', hue='value')
    plt.ylabel('total')
    plt.subplot(1, 2, 2)
    plt.title('cardio = 1')
    sns.countplot(data=cardio, x='variable', hue='value')
    plt.ylabel('total')
    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data

    df_heat = df.loc[(df['ap_lo'] <= df['ap_hi'])
                     & (df['height'] >= df['height'].quantile(0.025)) &
                     (df['height'] <= df['height'].quantile(0.975)) &
                     (df['weight'] >= df['weight'].quantile(0.025)) &
                     (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig = plt.figure(figsize=(12, 12))
    plt.plot()

    # Draw the heatmap with 'sns.heatmap()'

    sns.heatmap(corr,
                center=0,
                vmin=-0.16,
                vmax=0.3,
                cbar_kws={"shrink": 0.5, 'ticks': [-0.08, 0.00, 0.08, 0.16, 0.24]},
                square=True,
                mask=mask,
                fmt='.1f',
                annot=True,
                linewidths=.3)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
