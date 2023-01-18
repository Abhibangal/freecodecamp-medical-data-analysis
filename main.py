from math import floor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('medical_examination.csv', header=0)
# convert height in cm to height in m
df['height_in_m'] = df['height'] / 100

# Add new column overweight
df['Overweight'] = round(df['weight'] / (df['height_in_m'] ** 2),1)

# Modify the column, overweight > 25 then 1 else 0
df.loc[df['Overweight'] <= 25, 'Overweight'] = 0
df.loc[df['Overweight'] > 25, 'Overweight'] = 1

# modify the column, cholesterol = 1 then 0 else 1
df.loc[df['cholesterol'] == 1,'cholesterol'] = 0
df.loc[df['cholesterol'] > 1,'cholesterol'] = 1

# modify the column, gluc = 1 then 0 else 1
df.loc[df['gluc'] == 1, 'gluc'] = 0
df.loc[df['gluc'] > 1, 'gluc'] = 1

cardio = pd.melt(df.loc[(df['cardio'] == 1)],
                 id_vars=['id'],
                 value_vars=['active', 'alco', 'cholesterol', 'gluc', 'Overweight', 'smoke'])

no_cardio = pd.melt(df.loc[(df['cardio'] == 0)], id_vars=['id'], value_vars=['active', 'alco', 'cholesterol', 'gluc',
                                                                             'Overweight', 'smoke'])
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('cardio = 1')
sns.countplot(no_cardio, x='variable', hue='value')
plt.ylabel('Total', size= 12)
plt.subplot(1, 2, 2)
plt.title('cardio = 0')
sns.countplot(cardio, x='variable', hue='value')
plt.ylabel('Total', size= 12)
fig.savefig('catplot.png')

fig1 = plt.figure(figsize=(12, 12))
plt.plot()
# sns.heatmap(df.corr(), cbar=True, square=False)
# plt.show()


df.drop(['height_in_m'], axis=1, inplace=True)
# convert height in cm to height in m
df_clean = df.loc[(df['ap_lo'] <= df['ap_hi']) &
                  (df['height'] >= df['height'].quantile(0.025)) &
                  (df['height'] <= df['height'].quantile(0.975)) &
                  (df['weight'] >= df['weight'].quantile(0.025)) &
                  (df['weight'] <= df['weight'].quantile(0.975))
]


corr = round(df_clean.corr(),1)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
print(df_clean)
print(corr)
sns.heatmap(corr, center=0, cbar=True, square=True, mask=mask, fmt='.1f', annot=True, linewidths=.3)
fig1.savefig('heatmap.png')