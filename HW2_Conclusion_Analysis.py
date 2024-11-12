# Correlation analysis
correlation_matrix = data.corr()
print(correlation_matrix['Energy_Consumption'].sort_values(ascending=False))

# Using Decision Tree to analyze feature importance
from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)
feature_importance = tree_model.feature_importances_

# Display feature importance
features_list = ['Temperature', 'Humidity']
importance_df = pd.DataFrame({
    'Feature': features_list,
    'Importance': feature_importance
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

# In this final step, By comparing R², MAE, and MSE for each model (Linear Regression, Decision Trees, Neural Networks), we can identify the best performing model.
# By analyzing feature importance and correlations, we can determine which features have the most influence on predicting energy consumption. For example, temperature and humidity might have the most impact.
