import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

class EnhancedImprovedRecommender:
    def __init__(self, train_data):
        self.train_data = train_data
        self.global_mean = train_data['rating'].mean()
        
        # Create matrices first
        self.create_matrices()
        
        # Calculate bias terms
        self.user_bias = self._calculate_user_bias()
        self.item_bias = self._calculate_item_bias()
        
        # Precompute rating counts
        self.user_rating_counts = train_data['user_id'].value_counts().to_dict()
        self.book_rating_counts = train_data['book_id'].value_counts().to_dict()
        
        # Calculate dynamic global mean threshold
        self.global_mean_threshold = self._calculate_global_mean_threshold()
    
    def _calculate_global_mean_threshold(self, threshold_user=5, threshold_book=5):
        """
        Calculate mean rating for users/books below threshold.
        This mean is used only for those cases where either user or book is below threshold.
        """
        # Get all ratings where either user or book is below threshold
        below_threshold_mask = (
            (self.train_data['user_id'].map(lambda x: self.user_rating_counts.get(x, 0) < threshold_user)) |
            (self.train_data['book_id'].map(lambda x: self.book_rating_counts.get(x, 0) < threshold_book))
        )
        
        below_threshold_ratings = self.train_data[below_threshold_mask]['rating']
        
        if len(below_threshold_ratings) == 0:
            return self.global_mean
        
        return below_threshold_ratings.mean()
    
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
        
        # Calculate global means for filling
        user_means = self.train_data.groupby('user_id')['rating'].mean()
        book_means = self.train_data.groupby('book_id')['rating'].mean()
        
        # Fill user matrix with user means
        for user_id in self.user_item_filled.index:
            user_mean = user_means.get(user_id, self.global_mean)
            self.user_item_filled.loc[user_id] = self.user_item_filled.loc[user_id].fillna(user_mean)
        
        # Fill item matrix with book means
        for book_id in self.item_user_filled.index:
            book_mean = book_means.get(book_id, self.global_mean)
            self.item_user_filled.loc[book_id] = self.item_user_filled.loc[book_id].fillna(book_mean)
        
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
                        threshold_user=5, threshold_book=5):
        """
        Enhanced predict_combined that uses global mean threshold for cold start cases
        """
        user_count = self.user_rating_counts.get(user_id, 0)
        book_count = self.book_rating_counts.get(book_id, 0)
        
        # Use global mean threshold if either user or book is below threshold
        if user_count < threshold_user or book_count < threshold_book:
            return self.global_mean_threshold
        
        # Get predictions if enough ratings
        user_pred = (self.predict_user_based(user_id, book_id, k=user_k) 
                    if user_count >= threshold_user else None)
        item_pred = (self.predict_item_based(user_id, book_id, k=item_k) 
                    if book_count >= threshold_book else None)
        
        # Fallback strategy
        if user_pred is None and item_pred is None:
            return self.global_mean_threshold
        elif user_pred is None:
            return item_pred
        elif item_pred is None:
            return user_pred
        
        # Calculate adaptive alpha based on rating counts
        user_weight = min(1, user_count / 10)  # Increases with more user ratings
        item_weight = min(1, book_count / 10)  # Increases with more book ratings
        
        adaptive_alpha = (alpha * user_weight) / (alpha * user_weight + (1 - alpha) * item_weight)
        
        return adaptive_alpha * user_pred + (1 - adaptive_alpha) * item_pred

def grid_search_parameters(train_data):
    """Find optimal parameters with enhanced parameter search"""
    print("Starting grid search...")
    
    # Split data for validation
    train_data_part, test_data_part = train_test_split(
        train_data, test_size=0.2, random_state=42
    )
    
    recommender = EnhancedImprovedRecommender(train_data_part)
    
    best_rmse = float('inf')
    best_params = None
    best_global_mean_threshold = None
    
    # Enhanced parameter grid
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    user_ks = [30, 50, 70]
    item_ks = [30, 50, 70]
    thresholds = [2, 3, 4, 5, 7, 10]  # Unified thresholds for both users and books
    
    total_combinations = len(alphas) * len(user_ks) * len(item_ks) * len(thresholds)
    
    print(f"Testing {total_combinations} parameter combinations...")
    
    for alpha in tqdm(alphas, desc="Alpha"):
        for user_k in user_ks:
            for item_k in item_ks:
                for threshold in thresholds:
                    # Update global mean threshold for current threshold value
                    recommender.global_mean_threshold = recommender._calculate_global_mean_threshold(
                        threshold_user=threshold, 
                        threshold_book=threshold
                    )
                    
                    y_true = []
                    y_pred = []
                    
                    for _, row in test_data_part.iterrows():
                        predicted_rating = recommender.predict_combined(
                            row['user_id'], row['book_id'],
                            alpha=alpha,
                            user_k=user_k,
                            item_k=item_k,
                            threshold_user=threshold,
                            threshold_book=threshold
                        )
                        
                        if not np.isnan(predicted_rating):
                            y_true.append(row['rating'])
                            y_pred.append(predicted_rating)
                    
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = {
                            'alpha': alpha,
                            'user_k': user_k,
                            'item_k': item_k,
                            'threshold': threshold
                        }
                        best_global_mean_threshold = recommender.global_mean_threshold
                        print(f"\nNew best parameters found:")
                        print(f"Alpha: {alpha}, User K: {user_k}, Item K: {item_k}, "
                              f"Threshold: {threshold}")
                        print(f"Global Mean Threshold: {best_global_mean_threshold:.4f}")
                        print(f"RMSE: {rmse:.4f}")
    
    return best_params, best_rmse, best_global_mean_threshold

def main():
    print("Loading data...")
    train_data = pd.read_csv('train.csv')
    
    print("Finding optimal parameters...")
    best_params, best_rmse, best_global_mean_threshold = grid_search_parameters(train_data)
    
    print("\nBest parameters found:")
    print(f"Alpha: {best_params['alpha']}")
    print(f"User K: {best_params['user_k']}")
    print(f"Item K: {best_params['item_k']}")
    print(f"Threshold: {best_params['threshold']}")
    print(f"Global Mean Threshold: {best_global_mean_threshold:.4f}")
    print(f"Best RMSE: {best_rmse:.4f}")
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    final_model = EnhancedImprovedRecommender(train_data)
    
    # Update global mean threshold with best threshold value
    final_model.global_mean_threshold = final_model._calculate_global_mean_threshold(
        threshold_user=best_params['threshold'],
        threshold_book=best_params['threshold']
    )
    
    # Generate predictions for test set
    print("Generating predictions for test set...")
    test_data = pd.read_csv('test.csv')
    predictions = []
    
    for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
        prediction = final_model.predict_combined(
            row['user_id'], row['book_id'],
            alpha=best_params['alpha'],
            user_k=best_params['user_k'],
            item_k=best_params['item_k'],
            threshold_user=best_params['threshold'],
            threshold_book=best_params['threshold']
        )
        predictions.append(prediction)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_data['id'],
        'rating': predictions
    })
    
    submission.to_csv('enhanced_submission.csv', index=False)
    print("Predictions saved to 'enhanced_submission.csv'")

if __name__ == "__main__":
    main()
