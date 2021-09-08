import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from utils import *

sns.set_theme(style='whitegrid')

# Raw Data
exp_types = ['limited', 'original']
datasets = ['NCBI-D', 'BC5CDR-D', 'BC5CDR-C', 'COMETA', 'MedMentions']
limiteds = [0.91233, 0.90881, 0.97594, 0.73438, 0.5375]
original = [0.91105, 0.90929, 0.9823, 0.74862, 0.54421 ]

limiteds = [a * 100 for a in limiteds]
original = [a * 100 for a in original]

df = pd.DataFrame()
flattened_datasets, flattend_exp_types, flatten_data = [], [], []

for ix, dataset in enumerate(datasets):
    # limited
    flattened_datasets.append(dataset)
    flattend_exp_types.append(exp_types[0])
    flatten_data.append(limiteds[ix])

    # original
    flattened_datasets.append(dataset)
    flattend_exp_types.append(exp_types[1])
    flatten_data.append(original[ix])

df['Dataset'] = flattened_datasets
df['Attention Scope'] = flattend_exp_types
df['Acc@1'] = flatten_data
sns.set(font_scale=1.2)
g = sns.catplot(
    data=df, kind="bar",
    x='Dataset', y='Acc@1', hue='Attention Scope',
    palette='magma', alpha=.8, height=6, legend=False,
)

#g.fig.set_size_inches(12,8)
g.fig.subplots_adjust(top=0.81,right=0.86)

flatten_nbs = flatten([limiteds, original])
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
plt.title('Peformance of SapBERT (Acc@1) when the attention scope is limited.')
plt.show()
