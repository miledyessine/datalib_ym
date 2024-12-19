import pandas as pd
import numpy as np
from datalib_ym import DataLoader, DataTransformer, StatisticalAnalysis, DataVisualizer, Regression

# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'A': np.random.rand(100),
    'B': np.random.rand(100),
    'C': np.random.randint(0, 5, 100)
})

# Initialize DataLib components
loader = DataLoader()
transformer = DataTransformer()
stats = StatisticalAnalysis()
visualizer = DataVisualizer()
regression = Regression()

# Save and load data
loader.data = data
loader.save_csv("sample_data.csv")
loaded_data = loader.load_csv("sample_data.csv")

print("Data loaded successfully:")
print(loaded_data.head())

# Transform data
normalized_data = transformer.normalize(loaded_data[['A', 'B']])
print("\nNormalized data:")
print(normalized_data.head())

# Handle missing values (introduce some NaNs for demonstration)
loaded_data.loc[0:5, 'A'] = np.nan
cleaned_data = transformer.handle_missing_values(loaded_data)
print("\nData after handling missing values:")
print(cleaned_data.head())

# Perform statistical analysis
mean_A = stats.calculate_mean(cleaned_data['A'])
correlation_AB = stats.calculate_correlation(cleaned_data['A'], cleaned_data['B'])

print(f"\nMean of A: {mean_A}")
print(f"Correlation between A and B: {correlation_AB}")

# Visualize data
visualizer.plot_histogram(cleaned_data, 'A', title='Distribution of A')
visualizer.plot_scatter(cleaned_data, 'A', 'B', title='A vs B Scatter Plot')

# Perform regression
X = cleaned_data[['A', 'B']]
y = cleaned_data['C']
model, mse = regression.linear_regression(X, y)

print(f"\nLinear Regression MSE: {mse}")
print("Regression Coefficients:")
print(model.coef_)

# Save the correlation matrix plot
visualizer.plot_correlation_matrix(cleaned_data, title='Correlation Matrix')
print("\nCorrelation matrix plot saved.")

print("\nDataLib demo completed. Check the generated plots.")

