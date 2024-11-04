import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Intended to run separately from main. Generate graphs from experiment's output data

output_dir = '../output'
df = pd.read_csv('../output/outputs.csv')

sns.set_theme(style="whitegrid")
sns.set_palette("husl")

###################################
# Classification
###################################

plt.figure(figsize=(10, 6))
classification_data = df[df['accuracy'].notna()]

sns.barplot(x='dataset', y='accuracy', hue='hidden_layers', data=classification_data)

plt.title('Classification Accuracy by Dataset and Number of Hidden Layers', pad=20)
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.legend(title='Hidden Layers')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, 'classification_performance.png'), dpi=300, bbox_inches='tight')
plt.close()

###################################
# Regression
###################################

regression_data = df[df['mse'].notna()]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['mse', 'rmse', 'mae']
titles = ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    sns.barplot(x='dataset', y=metric, hue='hidden_layers', data=regression_data, ax=axes[i])
    axes[i].set_title(title)
    axes[i].set_xlabel('Dataset')
    axes[i].set_ylabel(metric.upper())
    axes[i].tick_params(axis='x', rotation=45)

    if regression_data[metric].max() > 100:  # log transform mse, hardware skews scaling
        axes[i].set_yscale('log')
        axes[i].set_ylabel(f'{metric.upper()} (log scale)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'regression_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()
