import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from utils import *

sns.set_theme(style='whitegrid')

# Raw Data
#exp_types = ['unigrams', 'bigrams', 'original']
exp_types = ['shuffled', 'original']
datasets = ['NCBI-D', 'BC5CDR-D', 'BC5CDR-C', 'COMETA', 'MedMentions']
unigrams = [0.88183, 0.90187, 0.94035, 0.65579, 0.53234]
#bigrams = [0.89072, 0.90833, 0.96381, 0.71875, 0.5379]
original = [0.91105, 0.90929, 0.9823, 0.74862, 0.54421 ]

unigrams = [a * 100 for a in unigrams]
#bigrams = [a * 100 for a in bigrams]
original = [a * 100 for a in original]

df = pd.DataFrame()
flattened_datasets, flattend_exp_types, flatten_data = [], [], []

for ix, dataset in enumerate(datasets):
    # unigram
    flattened_datasets.append(dataset)
    flattend_exp_types.append(exp_types[0])
    flatten_data.append(unigrams[ix])
    # bigram
    # flattened_datasets.append(dataset)
    # flattend_exp_types.append(exp_types[1])
    # flatten_data.append(bigrams[ix])
    # original
    flattened_datasets.append(dataset)
    flattend_exp_types.append(exp_types[1])
    flatten_data.append(original[ix])

df['Dataset'] = flattened_datasets
df['Words Order'] = flattend_exp_types
df['Acc@1'] = flatten_data
sns.set(font_scale=1.2)
g = sns.catplot(
    data=df, kind="bar",
    x='Dataset', y='Acc@1', hue='Words Order',
    palette='autumn', alpha=.8, height=6, legend=False,
)

#g.fig.set_size_inches(12,8)
g.fig.subplots_adjust(top=0.81,right=0.86)

#flatten_nbs = flatten([unigrams, bigrams, original])
flatten_nbs = flatten([unigrams, original])
ax = g.facet_axis(0,0)
for ix, p in enumerate(ax.patches):
    ax.text(p.get_x() + 0.05,
            p.get_height() - 5,
           '{0:.1f}'.format(flatten_nbs[ix]),   #Used to format it K representation
            color='black',
            rotation='horizontal',
            size='small')


g.set_axis_labels('', '')
plt.legend(loc='upper right')
plt.title('Peformance of SapBERT (Acc@1) when the input tokens are shuffled.', pad=25)
plt.show()
