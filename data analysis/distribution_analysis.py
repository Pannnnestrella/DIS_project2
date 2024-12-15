import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for better visualizations
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Load data
ratings = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')
book_features = pd.read_csv('./data/books_features.csv')
user_features = pd.read_csv('./data/user_features.csv')

def plot_feature_distributions(train_data, test_data, feature_name, title, bins=30):
    plt.figure(figsize=(12, 6))
    
    # Plot distributions
    plt.hist(train_data, bins=bins, alpha=0.5, label='Train', density=True)
    plt.hist(test_data, bins=bins, alpha=0.5, label='Test', density=True)
    
    # Add KDE curves
    sns.kdeplot(data=train_data, color='blue', linewidth=2, label='Train KDE')
    sns.kdeplot(data=test_data, color='orange', linewidth=2, label='Test KDE')
    
    # Calculate KS test
    ks_stat, p_value = stats.ks_2samp(train_data, test_data)
    
    plt.title(f'{title}\nKS test: statistic={ks_stat:.3f}, p-value={p_value:.3e}')
    plt.xlabel(feature_name)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return ks_stat, p_value

# Create directory for saving plots
import os
os.makedirs('distribution_plots', exist_ok=True)

# Analyze user features
print("Analyzing user features...")
train_users = ratings['user_id'].unique()
test_users = test_data['user_id'].unique()

train_user_features = user_features[user_features['user_id'].isin(train_users)]
test_user_features = user_features[user_features['user_id'].isin(test_users)]

user_numeric_features = ['num_ratings', 'avg_rating', 'std_rating', 'max_rating', 'min_rating']

for feature in user_numeric_features:
    ks_stat, p_value = plot_feature_distributions(
        train_user_features[feature],
        test_user_features[feature],
        feature,
        f'User {feature.replace("_", " ").title()} Distribution'
    )
    plt.savefig(f'distribution_plots/user_{feature}_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

# Analyze book features
print("\nAnalyzing book features...")
train_books = ratings['book_id'].unique()
test_books = test_data['book_id'].unique()

train_book_features = book_features[book_features['book_id'].isin(train_books)]
test_book_features = book_features[book_features['book_id'].isin(test_books)]

book_numeric_features = ['num_ratings', 'avg_rating', 'std_rating', 'max_rating', 'min_rating', 
                        'avg_category_rating', 'avg_publisher_rating']

for feature in book_numeric_features:
    ks_stat, p_value = plot_feature_distributions(
        train_book_features[feature],
        test_book_features[feature],
        feature,
        f'Book {feature.replace("_", " ").title()} Distribution'
    )
    plt.savefig(f'distribution_plots/book_{feature}_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

# Analyze categorical features
def plot_categorical_distribution(train_data, test_data, feature_name):
    plt.figure(figsize=(15, 6))
    
    # Calculate proportions for train and test
    train_props = train_data[feature_name].value_counts(normalize=True)
    test_props = test_data[feature_name].value_counts(normalize=True)
    
    # Get all unique categories
    all_categories = sorted(set(train_props.index) | set(test_props.index))
    
    # Create bar positions
    x = np.arange(len(all_categories))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, [train_props.get(cat, 0) for cat in all_categories], 
            width, label='Train', alpha=0.7)
    plt.bar(x + width/2, [test_props.get(cat, 0) for cat in all_categories], 
            width, label='Test', alpha=0.7)
    
    plt.xlabel(feature_name)
    plt.ylabel('Proportion')
    plt.title(f'{feature_name} Distribution in Train vs Test')
    plt.xticks(x, all_categories, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Chi-square test
    train_counts = pd.Series([len(train_data) * train_props.get(cat, 0) for cat in all_categories])
    test_counts = pd.Series([len(test_data) * test_props.get(cat, 0) for cat in all_categories])
    
    chi2, p_value = stats.chi2_contingency([train_counts, test_counts])[:2]
    plt.title(f'{feature_name} Distribution\nChi-square test: statistic={chi2:.3f}, p-value={p_value:.3e}')
    
    return chi2, p_value

# Analyze categorical features
for feature in ['categories', 'publisher']:
    chi2, p_value = plot_categorical_distribution(
        train_book_features,
        test_book_features,
        feature
    )
    plt.savefig(f'distribution_plots/book_{feature}_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

# Print summary statistics
print("\nSummary Statistics:")
print("\nUser Features:")
for feature in user_numeric_features:
    print(f"\n{feature}:")
    print("Train:", train_user_features[feature].describe())
    print("Test:", test_user_features[feature].describe())

print("\nBook Features:")
for feature in book_numeric_features:
    print(f"\n{feature}:")
    print("Train:", train_book_features[feature].describe())
    print("Test:", test_book_features[feature].describe())

# Calculate overlap statistics
user_overlap = len(set(train_users) & set(test_users))
book_overlap = len(set(train_books) & set(test_books))

print("\nOverlap Statistics:")
print(f"Users in train: {len(train_users)}, Users in test: {len(test_users)}")
print(f"User overlap: {user_overlap} ({user_overlap/len(train_users)*100:.2f}% of train, {user_overlap/len(test_users)*100:.2f}% of test)")
print(f"Books in train: {len(train_books)}, Books in test: {len(test_books)}")
print(f"Book overlap: {book_overlap} ({book_overlap/len(train_books)*100:.2f}% of train, {book_overlap/len(test_books)*100:.2f}% of test)")
