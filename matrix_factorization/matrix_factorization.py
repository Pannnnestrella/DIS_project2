import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import itertools
import os
import json
from datetime import datetime

def load_and_split_data(train_path, test_path, valid_size=0.2, random_state=42):
    """Load training and test data, and split training data into train and validation sets."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Split train data into train and validation
    train_data, valid_data = train_test_split(train_df, test_size=valid_size, random_state=random_state)
    
    return train_data, valid_data, test_df

def create_user_item_matrix(df):
    """Create a user-item matrix from dataframe."""
    # fill missing values with mean of column
    df.fillna(df.mean(), inplace=True)
    return df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)

def matrix_factorization(R, K=10, alpha=0.001, beta=0.02, iterations=100, batch_size=1000, early_stopping_rounds=5, min_improvement=0.0001):
    """
    Implement matrix factorization using gradient descent with optimizations.
    
    Args:
        R: Rating matrix
        K: Number of latent features
        alpha: Learning rate
        beta: Regularization parameter
        iterations: Maximum number of iterations
        batch_size: Size of mini-batches
        early_stopping_rounds: Number of rounds without improvement before stopping
        min_improvement: Minimum improvement in error to continue training
    """
    M, N = R.shape
    P = np.random.normal(scale=1./K, size=(M, K))
    Q = np.random.normal(scale=1./K, size=(N, K))
    
    # Initialize biases
    bu = np.zeros(M)
    bi = np.zeros(N)
    
    # Calculate global mean
    global_mean = np.mean(R[R != 0])
    
    # Get non-zero rating indices
    user_idx, item_idx = np.nonzero(R)
    n_ratings = len(user_idx)
    
    # Early stopping variables
    best_rmse = float('inf')
    rounds_without_improvement = 0
    rmse_history = []
    
    for it in range(iterations):
        # Shuffle indices for mini-batch processing
        indices = np.random.permutation(n_ratings)
        squared_error_sum = 0
        n_predictions = 0
        
        # Process in mini-batches
        for batch_start in range(0, n_ratings, batch_size):
            batch_indices = indices[batch_start:min(batch_start + batch_size, n_ratings)]
            batch_users = user_idx[batch_indices]
            batch_items = item_idx[batch_indices]
            
            # Calculate predicted ratings for batch
            batch_P = P[batch_users]
            batch_Q = Q[batch_items]
            batch_bu = bu[batch_users]
            batch_bi = bi[batch_items]
            
            # Vectorized prediction
            pred = (global_mean + 
                   batch_bu + 
                   batch_bi + 
                   np.sum(batch_P * batch_Q, axis=1))
            
            # Calculate error
            batch_R = R[batch_users, batch_items]
            errors_batch = batch_R - pred
            squared_error_sum += np.sum(errors_batch ** 2)
            n_predictions += len(batch_indices)
            
            # Vectorized updates
            for idx in range(len(batch_indices)):
                u = batch_users[idx]
                i = batch_items[idx]
                err = errors_batch[idx]
                
                # Update latent factors
                P[u] += alpha * (err * Q[i] - beta * P[u])
                Q[i] += alpha * (err * P[u] - beta * Q[i])
                
                # Update biases
                bu[u] += alpha * (err - beta * bu[u])
                bi[i] += alpha * (err - beta * bi[i])
        
        # Calculate RMSE including regularization
        reg_error = beta * (np.sum(P**2) + np.sum(Q**2) + np.sum(bu**2) + np.sum(bi**2))
        rmse = np.sqrt((squared_error_sum + reg_error) / n_predictions)
        rmse_history.append(rmse)
        
        # Early stopping check
        if rmse < best_rmse - min_improvement:
            best_rmse = rmse
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
        
        if rounds_without_improvement >= early_stopping_rounds:
            print(f"Early stopping at iteration {it+1}")
            break
        
        if (it + 1) % 10 == 0:
            print(f"Iteration {it+1}: RMSE = {rmse:.4f}")
    
    return P, Q, bu, bi, global_mean, rmse_history

def predict_rating(user_features, item_features, user_bias, item_bias, global_mean):
    """Predict rating given user and item features."""
    return global_mean + user_bias + item_bias + np.dot(user_features, item_features.T)

def calculate_rmse(true_ratings, predicted_ratings):
    """Calculate Root Mean Square Error."""
    return np.sqrt(mean_squared_error(true_ratings, predicted_ratings))

def grid_search(train_matrix, valid_df, param_grid):
    """
    Perform grid search to find best parameters.
    """
    results = []
    best_rmse = float('inf')
    best_params = None
    
    # Generate all parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    total_combinations = len(param_combinations)
    
    print(f"\nStarting grid search with {total_combinations} parameter combinations...")
    
    for i, params in enumerate(param_combinations, 1):
        print(f"\nTrying combination {i}/{total_combinations}:")
        print(params)
        
        # Train model with current parameters
        P, Q, bu, bi, global_mean, training_errors = matrix_factorization(
            train_matrix.values,
            **params
        )
        
        # Make predictions on validation set
        valid_predictions = []
        valid_true_ratings = []
        
        for idx, row in valid_df.iterrows():
            if row['user_id'] in train_matrix.index and row['book_id'] in train_matrix.columns:
                user_idx = train_matrix.index.get_loc(row['user_id'])
                item_idx = train_matrix.columns.get_loc(row['book_id'])
                
                pred = predict_rating(
                    P[user_idx], 
                    Q[item_idx].reshape(1, -1), 
                    bu[user_idx], 
                    bi[item_idx], 
                    global_mean
                )
                valid_predictions.append(pred)
                valid_true_ratings.append(row['rating'])
        
        # Calculate validation RMSE
        valid_rmse = calculate_rmse(valid_true_ratings, valid_predictions)
        print(f"Validation RMSE: {valid_rmse:.4f}")
        
        # Store results
        result = {
            **params,
            'valid_rmse': valid_rmse,
            'final_train_rmse': training_errors[-1]
        }
        results.append(result)
        
        # Update best parameters if needed
        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            best_params = params.copy()
            print("New best model found!")
    
    return best_params, results

def plot_parameter_comparison(results, save_dir):
    """Plot comparison of parameters and their effects on RMSE."""
    results_df = pd.DataFrame(results)
    params_to_plot = ['K', 'alpha', 'beta', 'batch_size']
    
    for param in params_to_plot:
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(results_df[param], results_df['valid_rmse'], alpha=0.6)
            plt.plot(results_df[param], results_df['valid_rmse'], 'r--', alpha=0.3)
            plt.xlabel(param)
            plt.ylabel('Validation RMSE')
            plt.title(f'Effect of {param} on Validation RMSE')
            plt.grid(True)
            save_path = os.path.join(save_dir, f'param_comparison_{param}.png')
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error plotting parameter {param}: {str(e)}")

def plot_rating_distribution(df, save_path=None):
    """Plot distribution of ratings."""
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(df['rating'], bins=20, edgecolor='black')
        plt.title('Distribution of Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            plt.close()
    except Exception as e:
        print(f"Error in plot_rating_distribution: {str(e)}")
        plt.close()

def plot_training_error(errors, save_path=None):
    """Plot training error over iterations."""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(errors)
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.title('Training RMSE vs. Iteration')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            plt.close()
    except Exception as e:
        print(f"Error in plot_training_error: {str(e)}")
        plt.close()

def main():
    # Create necessary directories
    os.makedirs('./img', exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    # 1. Load and prepare data
    print("Loading and splitting data...")
    train_df, valid_df, test_df = load_and_split_data('./data/train.csv', './data/test.csv')
    print(f"Train data shape: {train_df.shape}")
    print(f"Validation data shape: {valid_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    # 2. Analyze training data distribution
    print("\nAnalyzing rating distribution...")
    plot_rating_distribution(train_df, './img/rating_distribution.png')

    # 3. Create user-item matrices
    print("\nCreating user-item matrices...")
    train_matrix = create_user_item_matrix(train_df)

    # 4. Define parameter grid
    param_grid = {
        'K': [10, 20, 30],
        'alpha': [0.001, 0.005, 0.01],
        'beta': [0.01, 0.02, 0.05],
        'iterations': [50],
        'batch_size': [2000, 4000],
        'early_stopping_rounds': [3],
        'min_improvement': [0.001]
    }

    # 5. Perform grid search
    best_params, results = grid_search(train_matrix, valid_df, param_grid)
    
    # 6. Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'./results/grid_search_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'best_params': best_params,
            'all_results': results
        }, f, indent=4)
    print(f"\nSaved grid search results to {results_file}")

    # 7. Plot parameter comparisons
    print("\nPlotting parameter comparisons...")
    plot_parameter_comparison(results, './img')

    # 8. Train final model with best parameters
    print("\nTraining final model with best parameters...")
    P, Q, bu, bi, global_mean, training_errors = matrix_factorization(
        train_matrix.values,
        **best_params
    )

    # 9. Plot final training error
    print("\nPlotting training error...")
    plot_training_error(training_errors, './img/final_training_error.png')

    # 10. Make predictions on test set
    print("\nMaking predictions on test set...")
    test_predictions = []
    test_user_book_pairs = []
    
    for idx, row in test_df.iterrows():
        if row['user_id'] in train_matrix.index and row['book_id'] in train_matrix.columns:
            user_idx = train_matrix.index.get_loc(row['user_id'])
            item_idx = train_matrix.columns.get_loc(row['book_id'])
            
            pred = predict_rating(
                P[user_idx], 
                Q[item_idx].reshape(1, -1), 
                bu[user_idx], 
                bi[item_idx], 
                global_mean
            )
            test_predictions.append(pred)
            test_user_book_pairs.append((row['user_id'], row['book_id']))

    # 11. Create submission dataframe
    submission_df = pd.DataFrame({
        'user_id': [pair[0] for pair in test_user_book_pairs],
        'book_id': [pair[1] for pair in test_user_book_pairs],
        'rating': test_predictions
    })
    
    # Save predictions
    predictions_file = f'predictions_{timestamp}.csv'
    submission_df.to_csv(predictions_file, index=False)
    print(f"\nSaved test predictions to {predictions_file}")

if __name__ == "__main__":
    main()
