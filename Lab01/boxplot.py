import pandas as pd
import matplotlib.pyplot as plt 

_dir = './heart-replacemissing.csv'
df = pd.read_csv(_dir)

atrs = ['chol', 'oldpeak', 'ca']
for atr in atrs:
    plt.subplot(1, len(atrs), 1 + atrs.index(atr))
    plt.boxplot(df[atr])
    plt.xlabel(atr)
plt.show()