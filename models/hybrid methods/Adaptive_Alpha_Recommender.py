import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from itertools import product

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

def predict_combined_with_adaptive(
    user_id, book_id, user_item_matrix, user_similarity_df, item_similarity_df, 
    user_rating_counts, book_rating_counts, global_mean, alpha=0.5
):
    """
    Predict rating using a combination of user-based and item-based collaborative filtering
    with adaptive weighting based on rating counts
    """
    threshold = 2  # 固定阈值为2
    user_ratings_count = user_rating_counts.get(user_id, 0)
    book_ratings_count = book_rating_counts.get(book_id, 0)

    # 如果评分数都低于阈值，返回全局平均分
    if user_ratings_count < threshold and book_ratings_count < threshold:
        return global_mean

    # 获取两种预测方法的结果
    user_based_prediction = (
        predict_user_based_rating(user_id, book_id, user_item_matrix, user_similarity_df)
        if book_ratings_count >= threshold
        else np.nan
    )

    item_based_prediction = (
        predict_item_based_rating(user_id, book_id, user_item_matrix, item_similarity_df)
        if user_ratings_count >= threshold
        else np.nan
    )

    # 如果两种预测都失败，返回全局平均分
    if np.isnan(user_based_prediction) and np.isnan(item_based_prediction):
        return global_mean
    elif np.isnan(user_based_prediction):
        return item_based_prediction
    elif np.isnan(item_based_prediction):
        return user_based_prediction
    
    # 计算自适应权重
    user_weight = min(1, user_ratings_count / 10)
    item_weight = min(1, book_ratings_count / 10)
    adaptive_alpha = (alpha * user_weight) / (alpha * user_weight + (1 - alpha) * item_weight)
    
    return adaptive_alpha * user_based_prediction + (1 - adaptive_alpha) * item_based_prediction

def evaluate_parameters(user_item_matrix_train, user_similarity_df, item_similarity_df, 
                       test_data_part, user_rating_counts, book_rating_counts, global_mean, alpha):
    """Evaluate the model with given parameters"""
    y_true = []
    y_pred = []
    
    for _, row in test_data_part.iterrows():
        predicted_rating = predict_combined_with_adaptive(
            row['user_id'], row['book_id'], 
            user_item_matrix_train, user_similarity_df, item_similarity_df,
            user_rating_counts, book_rating_counts, global_mean,
            alpha=alpha
        )
        
        if not np.isnan(predicted_rating):
            y_true.append(row['rating'])
            y_pred.append(predicted_rating)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def grid_search_parameters(train_data):
    """Perform grid search to find optimal alpha"""
    print("Preparing data and calculating matrices...")
    
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
    
    # Define parameter grid (只搜索alpha)
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    best_rmse = float('inf')
    best_alpha = None
    
    print("Starting alpha search...")
    for alpha in tqdm(alphas, desc="Testing alpha values"):
        rmse = evaluate_parameters(
            user_item_matrix_train, user_similarity_df, item_similarity_df,
            test_data_part, user_rating_counts, book_rating_counts, 
            global_mean, alpha
        )
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
            print(f"\nNew best alpha found:")
            print(f"Alpha: {alpha}")
            print(f"RMSE: {rmse}")
    
    return best_alpha, best_rmse

if __name__ == "__main__":
    print("Starting parameter optimization...")
    train_data = pd.read_csv('train.csv')
    best_alpha, best_rmse = grid_search_parameters(train_data)
    
    print("\nOptimization complete!")
    print(f"Best alpha found: {best_alpha}")
    print(f"Best RMSE: {best_rmse}")
