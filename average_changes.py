datasets = ['NCBI-D', 'BC5CDR-D', 'BC5CDR-C', 'COMETA', 'MedMentions']

# Word ordering
unigrams = [0.88183, 0.90187, 0.94035, 0.65579, 0.53234]
bigrams = [0.89072, 0.90833, 0.96381, 0.71875, 0.5379]
trigrams = [0.9047, 0.90953, 0.97693, 0.73116, 0.53991]

# Limited
limiteds_3 = [0.91105, 0.90306, 0.97932, 0.71875, 0.53223]
limiteds_5 = [0.91233, 0.90881, 0.97594, 0.73438, 0.5375]

# Original
original = [0.91105, 0.90929, 0.9823, 0.74862, 0.54421]

#
exps = [unigrams, bigrams, trigrams, limiteds_3, limiteds_5]
exp_names = ['unigrams', 'bigrams', 'trigrams', 'limiteds_3', 'limiteds_5']

for exp, exp_name in zip(exps, exp_names):
    nb_datasets = len(datasets)
    total_changes = 0.0
    for j in range(nb_datasets):
        total_changes += (exp[j] - original[j]) / original[j]
    total_changes /= len(datasets)
    print(f'For {exp_name} | Average % change: {round(total_changes * 100, 2)}')
