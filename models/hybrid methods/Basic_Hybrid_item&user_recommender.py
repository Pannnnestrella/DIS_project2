import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from itertools import product

# Load the training data
train_data = pd.read_csv('train.csv')  # Updated path to be relative

def create_matrices(train_data_part):
    """Create and prepare matrices for collaborative filtering"""
    # Create user-item rating matrices for the training set
    user_item_matrix_train = train_data_part.pivot(index='user_id', columns='book_id', values='rating')
    item_user_matrix_train = user_item_matrix_train.T

    # Fill NaN values with means
    user_item_filled_train = user_item_matrix_train.apply(lambda row: row.fillna(row.mean()), axis=1)
    item_user_filled_train = item_user_matrix_train.apply(lambda row: row.fillna(row.mean()), axis=1)

    # Calculate similarity matrices
    user_similarity = cosine_similarity(user_item_filled_train)
    item_similarity = cosine_similarity(item_user_filled_train)

    # Convert to DataFrames
    user_similarity_df = pd.DataFrame(user_similarity, 
                                    index=user_item_matrix_train.index, 
                                    columns=user_item_matrix_train.index)
    item_similarity_df = pd.DataFrame(item_similarity, 
                                    index=item_user_matrix_train.index, 
                                    columns=item_user_matrix_train.index)

    return (user_item_matrix_train, item_user_matrix_train, 
            user_similarity_df, item_similarity_df)

def predict_user_based_rating(user_id, book_id, user_item_matrix, user_similarity_df):
    """Predict rating based on user similarities"""
    if book_id not in user_item_matrix.columns:
        return np.nan
    
    user_ratings = user_item_matrix.loc[:, book_id]
    user_similarities = (user_similarity_df.loc[user_id] 
                        if user_id in user_similarity_df.index 
                        else pd.Series(0, index=user_item_matrix.index))
    
    rated_users = user_ratings[user_ratings.notnull()].index
    similarities = user_similarities[rated_users]
    ratings = user_ratings[rated_users]
    
    if len(rated_users) == 0 or np.sum(np.abs(similarities)) == 0:
        return np.nan
    
    return np.dot(similarities, ratings) / np.sum(np.abs(similarities))

def predict_item_based_rating(user_id, book_id, user_item_matrix, item_similarity_df):
    """Predict rating based on item similarities"""
    if user_id not in user_item_matrix.index:
        return np.nan
    
    item_ratings = user_item_matrix.loc[user_id]
    item_similarities = (item_similarity_df.loc[book_id] 
                        if book_id in item_similarity_df.index 
                        else pd.Series(0, index=user_item_matrix.columns))
    
    rated_items = item_ratings[item_ratings.notnull()].index
    similarities = item_similarities[rated_items]
    ratings = item_ratings[rated_items]
    
    if len(rated_items) == 0 or np.sum(np.abs(similarities)) == 0:
        return np.nan
    
    return np.dot(similarities, ratings) / np.sum(np.abs(similarities))

def predict_combined_with_threshold(
    user_id, book_id, user_item_matrix, user_similarity_df, item_similarity_df, 
    user_rating_counts, book_rating_counts, global_mean, alpha=0.5, 
    threshold_book=5, threshold_user=5
):
    """
    Predict rating using a combination of user-based and item-based collaborative filtering
    with thresholds and fallback strategies
    """
    user_ratings_count = user_rating_counts.get(user_id, 0)
    book_ratings_count = book_rating_counts.get(book_id, 0)

    # Fallback to global mean for cold start cases
    if user_ratings_count < threshold_user and book_ratings_count < threshold_book:
        return global_mean

    # Get predictions from both methods if applicable
    user_based_prediction = (
        predict_user_based_rating(user_id, book_id, user_item_matrix, user_similarity_df)
        if book_ratings_count >= threshold_book
        else np.nan
    )

    item_based_prediction = (
        predict_item_based_rating(user_id, book_id, user_item_matrix, item_similarity_df)
        if user_ratings_count >= threshold_user
        else np.nan
    )

    # Combine predictions with fallback strategy
    if np.isnan(user_based_prediction) and np.isnan(item_based_prediction):
        return global_mean
    elif np.isnan(user_based_prediction):
        return item_based_prediction
    elif np.isnan(item_based_prediction):
        return user_based_prediction
    
    return alpha * user_based_prediction + (1 - alpha) * item_based_prediction

def evaluate_parameters(user_item_matrix_train, user_similarity_df, item_similarity_df, 
                       test_data_part, user_rating_counts, book_rating_counts, global_mean, params):
    """
    Evaluate the model with given parameters, using pre-calculated matrices
    """
    alpha, threshold_book, threshold_user = params
    
    y_true = []
    y_pred = []
    
    for _, row in test_data_part.iterrows():
        predicted_rating = predict_combined_with_threshold(
            row['user_id'], row['book_id'], 
            user_item_matrix_train, user_similarity_df, item_similarity_df,
            user_rating_counts, book_rating_counts, global_mean,
            alpha=alpha, threshold_book=threshold_book, threshold_user=threshold_user
        )
        
        if not np.isnan(predicted_rating):
            y_true.append(row['rating'])
            y_pred.append(predicted_rating)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def grid_search_parameters(train_data):
    """
    Perform grid search to find optimal parameters
    """
    print("Preparing data and calculating matrices (this will be done only once)...")
    
    # Split data
    train_data_part, test_data_part = train_test_split(
        train_data, test_size=0.2, random_state=42
    )
    
    # Calculate matrices once
    (user_item_matrix_train, _, user_similarity_df, 
     item_similarity_df) = create_matrices(train_data_part)
    
    # Calculate these values once
    global_mean = user_item_matrix_train.mean().mean()
    user_rating_counts = train_data_part['user_id'].value_counts().to_dict()
    book_rating_counts = train_data_part['book_id'].value_counts().to_dict()
    
    # Define parameter grid
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    threshold_books = [2, 4, 7, 10]  # Based on book rating quartiles
    threshold_users = [1, 2, 5, 10]  # Based on user rating quartiles
    
    best_rmse = float('inf')
    best_params = None
    
    # Create all combinations of parameters
    param_combinations = list(product(alphas, threshold_books, threshold_users))
    
    print("Starting parameter search...")
    # Evaluate each combination using pre-calculated matrices
    for params in tqdm(param_combinations, desc="Evaluating parameter combinations"):
        rmse = evaluate_parameters(
            user_item_matrix_train, user_similarity_df, item_similarity_df,
            test_data_part, user_rating_counts, book_rating_counts, 
            global_mean, params
        )
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            print(f"\nNew best parameters found:")
            print(f"Alpha: {params[0]}, Book Threshold: {params[1]}, User Threshold: {params[2]}")
            print(f"RMSE: {rmse}")
    
    return best_params, best_rmse

if __name__ == "__main__":
    print("Starting parameter optimization...")
    best_params, best_rmse = grid_search_parameters(train_data)
    
    print("\nOptimization complete!")
    print(f"Best parameters found:")
    print(f"Alpha: {best_params[0]}")
    print(f"Book Threshold: {best_params[1]}")
    print(f"User Threshold: {best_params[2]}")
    print(f"Best RMSE: {best_rmse}")
