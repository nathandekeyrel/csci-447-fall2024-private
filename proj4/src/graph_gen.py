import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

regression_data = pd.read_csv('../outputs/regression_results.csv')
classification_data = pd.read_csv('../outputs/classification_results.csv')

plt.style.use('seaborn')
sns.set_palette("husl")

###################################
# Regression
###################################
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))  # plots side-by-side

# by algorithm
sns.barplot(data=regression_data, x='dataset', y='rmse', hue='algorithm', ax=ax1)
ax1.set_title('RMSE by Algorithm')
ax1.set_xlabel('Dataset')
ax1.set_ylabel('RMSE')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(title='Algorithm', bbox_to_anchor=(1.05, 1))

# by layer
sns.barplot(data=regression_data, x='dataset', y='rmse', hue='layers', ax=ax2)
ax2.set_title('RMSE by Network Layers')
ax2.set_xlabel('Dataset')
ax2.set_ylabel('RMSE')
ax2.tick_params(axis='x', rotation=45)
ax2.legend(title='Hidden Layers', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig('../images/regression_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

###################################
# Classification
###################################
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))  # plot side-by-side

# by algorithm
sns.barplot(data=classification_data, x='dataset', y='accuracy', hue='algorithm', ax=ax1)
ax1.set_title('Accuracy by Algorithm')
ax1.set_xlabel('Dataset')
ax1.set_ylabel('Accuracy')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(title='Algorithm', bbox_to_anchor=(1.05, 1))

# by layers
sns.barplot(data=classification_data, x='dataset', y='accuracy', hue='layers', ax=ax2)
ax2.set_title('Accuracy by Network Layers')
ax2.set_xlabel('Dataset')
ax2.set_ylabel('Accuracy')
ax2.tick_params(axis='x', rotation=45)
ax2.legend(title='Hidden Layers', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig('../images/classification_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
