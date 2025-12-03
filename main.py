import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

users = pd.read_csv('users.csv')
books = pd.read_csv('books.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
genres = pd.read_csv('genres.csv')
book_genres = pd.read_csv('book_genres.csv')

book_genres = book_genres.merge(genres[['genre_id', 'books_count']], on='genre_id', how='left')
genre_stats = book_genres.groupby('book_id').agg({
    'genre_id': 'nunique',
    'books_count': 'mean'
}).reset_index()
genre_stats.columns = ['book_id', 'unique_genres_count', 'avg_genre_books_count']

train = train.merge(users, on='user_id', how='left')
train = train.merge(books, on='book_id', how='left')
train = train.merge(genre_stats, on='book_id', how='left')

test = test.merge(users, on='user_id', how='left')
test = test.merge(books, on='book_id', how='left')
test = test.merge(genre_stats, on='book_id', how='left')

full_data = pd.concat([train, test], ignore_index=True)

label_cols = ['gender', 'language', 'author_name', 'publisher']
for col in label_cols:
    le = LabelEncoder()
    full_data[col] = le.fit_transform(full_data[col].astype(str))

train = full_data[full_data['rating'].notna()].copy()
test = full_data[full_data['rating'].isna()].copy()

features = ['user_id', 'book_id', 'has_read', 'gender', 'age', 'author_id',
            'publication_year', 'language', 'avg_rating', 'author_name', 
            'publisher', 'unique_genres_count', 'avg_genre_books_count']

X = train[features]
y = train['rating']
X_test = test[features]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'n_estimators': 4500,
    'learning_rate': 0.06,
    'max_depth': 9,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_pred_clipped = np.clip(val_pred, 0, 10)
rmse = np.sqrt(mean_squared_error(y_val, val_pred_clipped))
mae = mean_absolute_error(y_val, val_pred_clipped)
score = 1 - (0.5 * (rmse / 10) + 0.5 * (mae / 10))
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Score: {score:.4f}")

test_predictions = model.predict(X_test)
test_predictions_clipped = np.clip(test_predictions, 0, 10)
test['rating_predict'] = test_predictions_clipped
submission = test[['user_id', 'book_id', 'rating_predict']]
submission.to_csv('submission.csv', index=False)