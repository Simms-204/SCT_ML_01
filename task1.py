import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Extract relevant features
# For square footage, we'll use the total living area
# For bedrooms and bathrooms, we'll need to identify these columns from the data

# Create feature matrix X and target variable y
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']

# Prepare training data
X = train_df[features]
y = train_df['SalePrice']  # Target variable

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on validation set
val_predictions = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, val_predictions)

print('Model Performance:')
print(f'RMSE: ${rmse:,.2f}')
print(f'R2 Score: {r2:.3f}')

# Print feature coefficients
for feature, coef in zip(features, model.coef_):
    print(f'{feature}: ${coef:,.2f}')

# Make predictions on test set
test_predictions = model.predict(test_df[features])

# Create submission file
submission = pd.DataFrame({
    'Id': test_df.index,
    'SalePrice': test_predictions
})
submission.to_csv('submission.csv', index=False)

