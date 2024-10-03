import json
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from wordcloud import WordCloud

# Modularization (Load data and preprocess data)
def load_data(file_path):
    reviews = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line)
            reviews.append(review)
    return pd.DataFrame(reviews)

def preprocess_data(df):
    df['reviewText'] = df['reviewText'].fillna('')
    return df

# Data Exploration
file_path = 'Luxury_Beauty_5.json.gz'
df = load_data(file_path)

# Preprocess the data
df = preprocess_data(df)

# Feature Engineering
df['reviewLength'] = df['reviewText'].apply(len)

# Select relevant features (including the new 'reviewLength')
selected_features = ['reviewText', 'reviewLength', 'overall']
df = df[selected_features]

# Split the data into training (60%) and testing (40%) sets
train_data, test_data = train_test_split(df, test_size=0.4, random_state=42)

# Vectorize the 'reviewText' feature using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = tfidf_vectorizer.fit_transform(train_data['reviewText'])
X_test = tfidf_vectorizer.transform(test_data['reviewText'])

# Use 'overall' as the target variable
y_train = train_data['overall']
y_test = test_data['overall']

# Hyperparameter Tuning with Ridge Regression
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_model = Ridge()
grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Print the best hyperparameter
print(f'Best Ridge Alpha: {grid_search.best_params_["alpha"]}')

# Train a Ridge regression model with the best alpha
best_alpha = grid_search.best_params_['alpha']
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train, y_train)

# Make predictions on the test set using Ridge Regression
ridge_predictions = ridge_model.predict(X_test)

# Evaluate the Ridge model
ridge_mse = mean_squared_error(y_test, ridge_predictions)
print(f'Ridge Mean Squared Error: {ridge_mse}')

# Train a Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions on the test set using Linear Regression
linear_predictions = linear_model.predict(X_test)

# Evaluate the Linear Regression model
linear_mse = mean_squared_error(y_test, linear_predictions)
print(f'Linear Regression Mean Squared Error: {linear_mse}')


print("\n" * 2)

# Plot a histogram of predicted ratings
plt.figure(figsize=(8, 6))
sns.histplot(ridge_predictions, bins=20, kde=True, color='skyblue', label='Ridge Predicted Ratings')
sns.histplot(linear_predictions, bins=20, kde=True, color='green', label='Linear Predicted Ratings')
sns.histplot(y_test, bins=20, kde=True, color='orange', label='Actual Ratings')
plt.title('Histogram of Predicted Ratings (Ridge vs Linear Regression)')
plt.xlabel('Ratings')
plt.ylabel('Frequency')
plt.legend()
plt.show()


print("\n" * 2)

# Generate and plot the word cloud
wordcloud = WordCloud(width=800, height=400, random_state=42, max_words=50, background_color='white').generate(' '.join(df['reviewText']))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Reviews')
plt.axis('off')
plt.show()
