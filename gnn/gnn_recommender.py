import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, to_hetero
from torch.nn import Sequential, ReLU, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class BookRecommenderGNN(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        
        # User feature transformation
        self.user_lin1 = Linear(-1, hidden_channels)
        self.user_lin2 = Linear(hidden_channels, hidden_channels)
        
        # Book feature transformation
        self.book_lin1 = Linear(-1, hidden_channels)
        self.book_lin2 = Linear(hidden_channels, hidden_channels)
        
        # Graph convolution layers
        self.convs = HeteroConv({
            ('user', 'rates', 'book'): SAGEConv((-1, -1), hidden_channels),
            ('book', 'rated_by', 'user'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')
        
        self.convs2 = HeteroConv({
            ('user', 'rates', 'book'): SAGEConv((-1, -1), hidden_channels),
            ('book', 'rated_by', 'user'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')
        
        # Rating prediction layers
        self.rating_predictor = Sequential(
            Linear(2 * hidden_channels, hidden_channels),
            ReLU(),
            Dropout(0.2),
            Linear(hidden_channels, 1)
        )
        
        self.dropout = Dropout(0.2)
        
    def forward(self, x_dict, edge_index_dict, edge_label_index=None):
        # Process user features with experience-based weights
        user_x = x_dict['user']
        num_ratings = user_x[:, -1:] # Last column is num_ratings
        
        # Apply different weights based on user experience
        exp_weights = torch.sigmoid(num_ratings / 10.0)  # Normalize experience
        
        # Initial user feature transformation with experience weighting
        user_h = self.user_lin1(user_x)
        user_h = F.relu(user_h)
        user_h = user_h * exp_weights  # Weight features by experience
        user_h = self.dropout(user_h)
        user_h = self.user_lin2(user_h)
        
        # Process book features
        book_h = self.book_lin1(x_dict['book'])
        book_h = F.relu(book_h)
        book_h = self.dropout(book_h)
        book_h = self.book_lin2(book_h)
        
        # First convolution layer
        x_dict = {
            'user': user_h,
            'book': book_h
        }
        
        x_dict = self.convs(x_dict, edge_index_dict)
        x_dict = {
            'user': F.relu(x_dict['user']) * exp_weights,  # Re-weight after convolution
            'book': F.relu(x_dict['book'])
        }
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        # Second convolution layer
        x_dict = self.convs2(x_dict, edge_index_dict)
        x_dict = {
            'user': F.relu(x_dict['user']) * exp_weights,  # Re-weight after convolution
            'book': F.relu(x_dict['book'])
        }
        
        # Use provided edge indices or default to the ones in edge_index_dict
        if edge_label_index is None:
            edge_label_index = edge_index_dict[('user', 'rates', 'book')]
        
        # Get embeddings for the edges we want to predict
        user_idx = edge_label_index[0]
        book_idx = edge_label_index[1]
        
        user_emb = x_dict['user'][user_idx]
        book_emb = x_dict['book'][book_idx]
        
        # Concatenate and predict
        x = torch.cat([user_emb, book_emb], dim=-1)
        pred = self.rating_predictor(x)
        
        # Scale predictions to be between 1 and 5
        pred = 1 + 4 * torch.sigmoid(pred)
        
        return pred.squeeze()

def load_and_preprocess_data():
    # Load all data
    ratings = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    book_features = pd.read_csv("data/books_features.csv")
    user_features = pd.read_csv("data/user_features.csv")

    # user_features = user_features[['user_id','num_ratings','min_rating','max_rating']]
    
    # Process book features
    numerical_features = ['avg_category_rating', 'avg_publisher_rating']
    categorical_features = ['publish_year','categories', 'publisher']
    # Encode categorical features
    for feature in categorical_features:
        book_features[feature] = LabelEncoder().fit_transform(book_features[feature])
    # Fill missing values with mean
    book_features[numerical_features] = book_features[numerical_features].fillna(book_features[numerical_features].mean())
    # Drop uncessary cols
    book_features = book_features.drop(['published_date'], axis=1)

    # Fillna
    user_features = user_features.apply(lambda x: x.fillna(x.mean()), axis=0)
    book_features = book_features.apply(lambda x: x.fillna(x.mean()), axis=0)

    # Remap user and book IDs to continuous range
    user_id_map = {id: i for i, id in enumerate(user_features['user_id'].unique())}
    book_id_map = {id: i for i, id in enumerate(book_features['book_id'].unique())}

    # Map IDs in ratings and test data
    ratings['user_id'] = ratings['user_id'].map(user_id_map)
    ratings['book_id'] = ratings['book_id'].map(book_id_map)
    test_data['user_id'] = test_data['user_id'].map(user_id_map)
    test_data['book_id'] = test_data['book_id'].map(book_id_map)

    # Reorder feature matrices according to the mapping
    user_features['user_id'] = user_features['user_id'].map(user_id_map)
    book_features['book_id'] = book_features['book_id'].map(book_id_map)
    user_features = user_features.sort_values('user_id')
    book_features = book_features.sort_values('book_id')

    # Convert features to tensors
    book_x = torch.tensor(book_features.drop(['book_id'], axis=1).values, dtype=torch.float)
    user_x = torch.tensor(user_features.drop(['user_id'], axis=1).values, dtype=torch.float)
    
    # Features for test data, num ratings + 1 for test data
    test_user_features = user_features.copy()
    test_user_features['num_ratings'] = test_user_features['num_ratings'] + 1
    test_user_features = test_user_features.sort_values('user_id')
    test_user_x = torch.tensor(test_user_features.drop(['user_id'], axis=1).values, dtype=torch.float)

    test_book_features = book_features.copy()
    test_book_features['num_ratings'] = test_book_features['num_ratings'] + 1
    test_book_features = test_book_features.sort_values('book_id')
    test_book_x = torch.tensor(test_book_features.drop(['book_id'], axis=1).values, dtype=torch.float)
    
    # Split training data into train and validation
    train_mask = np.random.rand(len(ratings)) < 0.8
    train_ratings = ratings[train_mask]
    val_ratings = ratings[~train_mask]
    
    # Create train and validation edge indices
    train_edge_index = torch.tensor([
        train_ratings['user_id'].values,
        train_ratings['book_id'].values
    ], dtype=torch.long)
    train_edge_label = torch.tensor(train_ratings['rating'].values, dtype=torch.float)
    
    val_edge_index = torch.tensor([
        val_ratings['user_id'].values,
        val_ratings['book_id'].values
    ], dtype=torch.long)
    val_edge_label = torch.tensor(val_ratings['rating'].values, dtype=torch.float)
    
    # Create test edge index
    test_edge_index = torch.tensor([
        test_data['user_id'].values,
        test_data['book_id'].values
    ], dtype=torch.long)
    
    # Create full training edge index and label
    full_edge_index = torch.tensor([
        ratings['user_id'].values,
        ratings['book_id'].values
    ], dtype=torch.long)
    full_edge_label = torch.tensor(ratings['rating'].values, dtype=torch.float)
    
    # Create training data with train/val split
    train_data = HeteroData()
    train_data['user'].x = user_x
    train_data['book'].x = book_x
    train_data['user', 'rates', 'book'].train_edge_index = train_edge_index
    train_data['user', 'rates', 'book'].train_edge_label = train_edge_label
    train_data['user', 'rates', 'book'].val_edge_index = val_edge_index
    train_data['user', 'rates', 'book'].val_edge_label = val_edge_label
    
    # Create test data
    test_data = HeteroData()
    test_data['user'].x = test_user_x
    test_data['book'].x = test_book_x
    test_data['user', 'rates', 'book'].edge_index = test_edge_index
    
    # Create full training data
    full_data = HeteroData()
    full_data['user'].x = user_x
    full_data['book'].x = book_x
    full_data['user', 'rates', 'book'].edge_index = full_edge_index
    full_data['user', 'rates', 'book'].edge_label = full_edge_label
    
    return train_data, test_data, full_data

def train_and_validate(model, data, optimizer, scheduler, epochs=350):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience = 20
    counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(data.x_dict, {
            ('user', 'rates', 'book'): data['user', 'rates', 'book'].train_edge_index,
            ('book', 'rated_by', 'user'): torch.stack([
                data['user', 'rates', 'book'].train_edge_index[1],
                data['user', 'rates', 'book'].train_edge_index[0]
            ])
        })
        
        target = data['user', 'rates', 'book'].train_edge_label
        loss = F.mse_loss(pred, target)
        
        # Add L2 regularization
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in model.parameters():
            l2_reg = l2_reg + torch.norm(param)
        
        loss = loss + 0.001 * l2_reg
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_losses.append(float(loss))

        # Validation
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(data.x_dict, {
                    ('user', 'rates', 'book'): data['user', 'rates', 'book'].val_edge_index,
                    ('book', 'rated_by', 'user'): torch.stack([
                        data['user', 'rates', 'book'].val_edge_index[1],
                        data['user', 'rates', 'book'].val_edge_index[0]
                    ])
                })
                
                val_target = data['user', 'rates', 'book'].val_edge_label
                val_loss = np.sqrt(mean_squared_error(val_target, val_pred))
                val_losses.append(float(val_loss))
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    best_epoch = epoch
                    counter = 0
                else:
                    counter += 1
                
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch: {epoch+1:03d}, Train Loss: {loss:.4f}, Val RMSE: {val_loss:.4f}')
    
    return best_val_loss, best_epoch, train_losses, val_losses

def train_final_model(model, data, optimizer, epochs=200):
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(data.x_dict, {
            ('user', 'rates', 'book'): data['user', 'rates', 'book'].edge_index,
            ('book', 'rated_by', 'user'): torch.stack([
                data['user', 'rates', 'book'].edge_index[1],
                data['user', 'rates', 'book'].edge_index[0]
            ])
        })
        
        loss = F.mse_loss(pred, data['user', 'rates', 'book'].edge_label)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    
    return train_losses

if __name__ == "__main__":
    print("Starting hyperparameter tuning...")
    
    # Load and preprocess data
    train_data, test_data, full_data = load_and_preprocess_data()
    
    # Create validation data
    tune_data = train_data.clone()
    
    # Hyperparameter configurations to try
    configs = [
        {'hidden_channels': 64, 'lr': 0.001, 'weight_decay': 0.01},
    ]
    
    # Find best hyperparameters
    best_loss = float('inf')
    best_config = None
    best_epoch = 0
    
    for config in configs:
        print(f"\nTrying config: {config}")
        model = BookRecommenderGNN(hidden_channels=config['hidden_channels'], 
                                 metadata=train_data.metadata())
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=config['lr'], 
                                    weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        val_loss, epochs, _, _ = train_and_validate(model, train_data, optimizer, scheduler)
        print(f"Config {config} - Val Loss: {val_loss:.4f}, Epochs: {epochs}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_config = config
            best_epoch = epochs
    
    print(f"\nBest config found: {best_config}")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Best number of epochs: {best_epoch}")
    
    # Train final model on all data with best config
    print("\nTraining final model on all data...")
    
    # Initialize final model with best config
    final_model = BookRecommenderGNN(hidden_channels=best_config['hidden_channels'],
                                   metadata=full_data.metadata())
    final_optimizer = torch.optim.AdamW(final_model.parameters(), 
                                      lr=best_config['lr'], 
                                      weight_decay=best_config['weight_decay'])
    
    # Train final model using full training data
    train_losses = train_final_model(final_model, full_data, final_optimizer, epochs=best_epoch)
    
    # Make predictions on test set
    final_model.eval()
    with torch.no_grad():
        test_pred = final_model(test_data.x_dict, {
            ('user', 'rates', 'book'): test_data['user', 'rates', 'book'].edge_index,
            ('book', 'rated_by', 'user'): torch.stack([
                test_data['user', 'rates', 'book'].edge_index[1],
                test_data['user', 'rates', 'book'].edge_index[0]
            ])
        })
    
    # Create prediction DataFrame
    predictions_df = pd.DataFrame({
        'id': [i for i in range(len(test_pred))],
        'rating': test_pred.numpy()
    })
    predictions_df.to_csv('predictions.csv', index=False)
    print("\nPredictions saved to predictions.csv")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.tight_layout()
    plt.show()

    # Plot rating distributions
    plt.figure(figsize=(12, 6))
    plt.hist(pd.read_csv("data/train.csv")['rating'], bins=50, alpha=0.5, label='Train', density=True)
    plt.hist(predictions_df['rating'], bins=100, alpha=0.5, label='Test', density=True)
    plt.title('Rating Distributions')
    plt.xlabel('Rating')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
