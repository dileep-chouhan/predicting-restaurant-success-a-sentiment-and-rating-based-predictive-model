import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_restaurants = 100
data = {
    'Restaurant_ID': range(1, num_restaurants + 1),
    'Avg_Rating': np.random.uniform(2.5, 4.5, num_restaurants),
    'Num_Reviews': np.random.randint(10, 500, num_restaurants),
    'Avg_Sentiment_Score': np.random.uniform(-1, 1, num_restaurants), # -1: very negative, 1: very positive
    'Price_Range': np.random.choice(['$', '$$', '$$$'], num_restaurants),
    'Cuisine_Type': np.random.choice(['Italian', 'Mexican', 'American', 'Asian'], num_restaurants),
    'Location_Rating': np.random.uniform(1, 5, num_restaurants)
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering (minimal for synthetic data) ---
#In a real-world scenario, this section would involve handling missing values, outliers, etc.
# --- 3. Analysis ---
#Correlation analysis
correlation_matrix = df[['Avg_Rating', 'Num_Reviews', 'Avg_Sentiment_Score', 'Location_Rating']].corr()
#Calculate the correlation between average rating and sentiment score
correlation, p_value = pearsonr(df['Avg_Rating'], df['Avg_Sentiment_Score'])
print(f"Correlation between Average Rating and Sentiment Score: {correlation:.2f} (p-value: {p_value:.3f})")
# --- 4. Visualization ---
#Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Restaurant Attributes')
plt.savefig('correlation_heatmap.png')
print("Plot saved to correlation_heatmap.png")
#Average Rating Distribution
plt.figure(figsize=(8,6))
sns.histplot(df['Avg_Rating'], kde=True)
plt.title('Distribution of Average Restaurant Ratings')
plt.savefig('rating_distribution.png')
print("Plot saved to rating_distribution.png")
#Scatter plot of Average Rating vs. Number of Reviews
plt.figure(figsize=(8,6))
sns.scatterplot(x='Num_Reviews', y='Avg_Rating', data=df)
plt.title('Average Rating vs. Number of Reviews')
plt.savefig('rating_vs_reviews.png')
print("Plot saved to rating_vs_reviews.png")
#Further analysis (e.g., regression modeling) would be done here in a full project.