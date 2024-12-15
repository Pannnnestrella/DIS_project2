import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

class ImprovedRecommender:
    def __init__(self, train_data):
        self.train_data = train_data
        self.global_mean = train_data['rating'].mean()
        
        # Calculate bias terms
        self.user_bias = self._calculate_user_bias()
        self.item_bias = self._calculate_item_bias()
        
        # Create matrices
        self.create_matrices()
        
        # Precompute rating counts
        self.user_rating_counts = train_data['user_id'].value_counts().to_dict()
        self.book_rating_counts = train_data['book_id'].value_counts().to_dict()
    
    def _calculate_user_bias(self):
        """Calculate how much each user deviates from the global mean"""
        user_means = self.train_data.groupby('user_id')['rating'].mean()
        return (user_means - self.global_mean).to_dict()
    
    def _calculate_item_bias(self):
        """Calculate how much each book deviates from the global mean"""
        book_means = self.train_data.groupby('book_id')['rating'].mean()
        return (book_means - self.global_mean).to_dict()
    
    def create_matrices(self):
        """Create user-item and item-user matrices with bias correction"""
        print("Creating matrices...")
        
        # Create base matrices
        self.user_item_matrix = self.train_data.pivot(
            index='user_id', columns='book_id', values='rating'
        )
        self.item_user_matrix = self.user_item_matrix.T
        
        # Fill NaN with bias-corrected values
        self.user_item_filled = self.user_item_matrix.copy()
        self.item_user_filled = self.item_user_matrix.copy()
        
        # Fill user matrix with user bias-corrected means
        for user_id in self.user_item_filled.index:
            user_bias = self.user_bias.get(user_id, 0)
            self.user_item_filled.loc[user_id] = self.user_item_filled.loc[user_id].fillna(
                self.global_mean + user_bias
            )
        
        # Fill item matrix with item bias-corrected means
        for book_id in self.item_user_filled.index:
            item_bias = self.item_bias.get(book_id, 0)
            self.item_user_filled.loc[book_id] = self.item_user_filled.loc[book_id].fillna(
                self.global_mean + item_bias
            )
        
        print("Calculating similarities...")
        # Calculate similarities
        user_similarity = cosine_similarity(self.user_item_filled)
        item_similarity = cosine_similarity(self.item_user_filled)
        
        # Convert to DataFrames
        self.user_similarity_df = pd.DataFrame(
            user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        self.item_similarity_df = pd.DataFrame(
            item_similarity,
            index=self.item_user_matrix.index,
            columns=self.item_user_matrix.index
        )
    
    def predict_user_based(self, user_id, book_id, k=50):
        """Predict rating using user-based CF with bias correction and top-k neighbors"""
        if book_id not in self.user_item_matrix.columns:
            return self.global_mean + self.user_bias.get(user_id, 0)
        
        user_ratings = self.user_item_matrix.loc[:, book_id]
        user_similarities = (self.user_similarity_df.loc[user_id] 
                           if user_id in self.user_similarity_df.index 
                           else pd.Series(0, index=self.user_item_matrix.index))
        
        # Get users who rated this book
        rated_users = user_ratings[user_ratings.notnull()].index
        similarities = user_similarities[rated_users]
        ratings = user_ratings[rated_users]
        
        if len(rated_users) == 0:
            return self.global_mean + self.user_bias.get(user_id, 0)
        
        # Use only top-k similar users
        if len(similarities) > k:
            top_k_idx = np.argsort(similarities)[-k:]
            similarities = similarities.iloc[top_k_idx]
            ratings = ratings.iloc[top_k_idx]
        
        # Remove bias from ratings for weighted average
        ratings_unbiased = ratings - [self.user_bias.get(u, 0) for u in ratings.index]
        
        if np.sum(np.abs(similarities)) == 0:
            return self.global_mean + self.user_bias.get(user_id, 0)
        
        # Add user bias back after weighted average
        prediction = (np.dot(similarities, ratings_unbiased) / 
                     np.sum(np.abs(similarities)) + 
                     self.user_bias.get(user_id, 0))
        
        return prediction
    
    def predict_item_based(self, user_id, book_id, k=50):
        """Predict rating using item-based CF with bias correction and top-k neighbors"""
        if user_id not in self.user_item_matrix.index:
            return self.global_mean + self.item_bias.get(book_id, 0)
        
        item_ratings = self.user_item_matrix.loc[user_id]
        item_similarities = (self.item_similarity_df.loc[book_id] 
                           if book_id in self.item_similarity_df.index 
                           else pd.Series(0, index=self.user_item_matrix.columns))
        
        # Get items rated by this user
        rated_items = item_ratings[item_ratings.notnull()].index
        similarities = item_similarities[rated_items]
        ratings = item_ratings[rated_items]
        
        if len(rated_items) == 0:
            return self.global_mean + self.item_bias.get(book_id, 0)
        
        # Use only top-k similar items
        if len(similarities) > k:
            top_k_idx = np.argsort(similarities)[-k:]
            similarities = similarities.iloc[top_k_idx]
            ratings = ratings.iloc[top_k_idx]
        
        # Remove bias from ratings for weighted average
        ratings_unbiased = ratings - [self.item_bias.get(b, 0) for b in ratings.index]
        
        if np.sum(np.abs(similarities)) == 0:
            return self.global_mean + self.item_bias.get(book_id, 0)
        
        # Add item bias back after weighted average
        prediction = (np.dot(similarities, ratings_unbiased) / 
                     np.sum(np.abs(similarities)) + 
                     self.item_bias.get(book_id, 0))
        
        return prediction
    
    def predict_combined(self, user_id, book_id, alpha=0.5, 
                        user_k=50, item_k=50, 
                        min_user_ratings=2, min_book_ratings=2):
        """
        Predict rating using a weighted combination of user-based and item-based CF
        with adaptive weighting based on number of ratings
        """
        user_count = self.user_rating_counts.get(user_id, 0)
        book_count = self.book_rating_counts.get(book_id, 0)
        
        # Get predictions if enough ratings
        user_pred = (self.predict_user_based(user_id, book_id, k=user_k) 
                    if user_count >= min_user_ratings else None)
        item_pred = (self.predict_item_based(user_id, book_id, k=item_k) 
                    if book_count >= min_book_ratings else None)
        
        # Adaptive weighting based on rating counts
        if user_pred is None and item_pred is None:
            return self.global_mean
        elif user_pred is None:
            return item_pred
        elif item_pred is None:
            return user_pred
        
        # Calculate adaptive alpha based on rating counts
        user_weight = min(1, user_count / 10)  # Increases with more user ratings
        item_weight = min(1, book_count / 10)  # Increases with more book ratings
        
        adaptive_alpha = (alpha * user_weight) / (alpha * user_weight + (1 - alpha) * item_weight)
        
        return adaptive_alpha * user_pred + (1 - adaptive_alpha) * item_pred

def grid_search_parameters(train_data, test_data):
    """Find optimal parameters"""
    print("Starting grid search...")
    
    recommender = ImprovedRecommender(train_data)
    
    best_rmse = float('inf')
    best_params = None
    
    # Parameter grid
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
    user_ks = [30, 50, 70]
    item_ks = [30, 50, 70]
    min_ratings = [2, 3, 5]
    
    total_combinations = len(alphas) * len(user_ks) * len(item_ks) * len(min_ratings)
    
    with tqdm(total=total_combinations) as pbar:
        for alpha in alphas:
            for user_k in user_ks:
                for item_k in item_ks:
                    for min_rating in min_ratings:
                        y_true = []
                        y_pred = []
                        
                        for _, row in test_data.iterrows():
                            prediction = recommender.predict_combined(
                                row['user_id'], row['book_id'],
                                alpha=alpha,
                                user_k=user_k,
                                item_k=item_k,
                                min_user_ratings=min_rating,
                                min_book_ratings=min_rating
                            )
                            
                            if not np.isnan(prediction):
                                y_true.append(row['rating'])
                                y_pred.append(prediction)
                        
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = {
                                'alpha': alpha,
                                'user_k': user_k,
                                'item_k': item_k,
                                'min_ratings': min_rating
                            }
                            print(f"\nNew best parameters found:")
                            print(f"Alpha: {alpha}")
                            print(f"User k: {user_k}")
                            print(f"Item k: {item_k}")
                            print(f"Min ratings: {min_rating}")
                            print(f"RMSE: {rmse}")
                        
                        pbar.update(1)
    
    return best_params, best_rmse

def main():
    # Load data
    print("Loading data...")
    train_data = pd.read_csv('train.csv')
    
    # Split data
    train_part, test_part = train_test_split(train_data, test_size=0.2, random_state=42)
    
    # Find optimal parameters
    best_params, best_rmse = grid_search_parameters(train_part, test_part)
    
    print("\nFinal Results:")
    print(f"Best parameters: {best_params}")
    print(f"Best RMSE: {best_rmse}")

if __name__ == "__main__":
    main()
