import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Load preprocessed data
df = pd.read_csv('preprocessed_data.csv')

# Split data into training and testing sets
X = df.drop(['Close_norm', 'Datetime'], axis=1)
y = df['Close_norm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on test set
y_pred = regressor.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print('Mean Absolute Error:', mae)
print('Root Mean Squared Error:', rmse)
