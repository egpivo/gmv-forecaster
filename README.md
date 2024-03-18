# GMV Forecasting

## Exploratory Data Analysis

### Summary:

1. **User Behavior Variability**: User behaviors exhibit variations across different time scales (e.g., hourly or monthly).

2. **Store Activity**: Approximately 60,000 stores have recorded transactions recently, indicating ongoing business activity. The most frequently visited store has been recorded 11 times, suggesting popularity among users.

3. **Spatial Relationship Analysis**: Utilizing K-means clustering, spatial locations are grouped to capture the original spatial relationship.

4. **Context Feature Exploration**: RMF analysis is employed to identify significant context features.

For detailed insights, refer to the [exploratory data analysis notebook](notebooks/exploratory_data_analysis.ipynb).

### Feature Engineering:

- **Missing Value Imputation**: Median imputation for numeric features and mode imputation for categorical features.

- **Context Feature Creation**: New context features are generated based on the findings from the exploratory data analysis notebook.

- **Binarization of Continuous Features**: Continuous features are binarized based on their summary statistics to simplify training of the xDeepFM model.

## Methodology

The forecasting of GMV is approached through two steps:

1. **Predicting Purchase Probability**: A Click-Through Rate (CTR) model is utilized to forecast the probability of a user making a purchase at a store for the next step.

2. **Estimating GMV**: Leveraging the estimated probability and the average store's GMV, the next GMV of either a user or a day is forecasted.

### Training Model: Extreme DeepFM (xDeepFM)

Extreme DeepFM (xDeepFM) is a powerful neural network architecture designed for tasks such as click-through rate prediction and recommendation systems. It extends the traditional DeepFM model by incorporating a Compressed Interaction Network (CIN) layer, which allows for capturing higher-order feature interactions more effectively.

#### Key Components:

- **Embedding Layer**: Converts categorical features into dense representations, facilitating meaningful feature interactions.

- **Deep Component**: Consists of multiple fully connected layers, enabling the model to learn complex patterns from the data.

- **Compressed Interaction Network (CIN)**: Captures feature interactions through a series of cross-network layers. Unlike traditional interaction methods, CIN utilizes a more efficient approach to compute feature interactions, making it suitable for large-scale datasets.

#### Training Process:

- **Positive and Negative Sampling**: The model is trained using both positive labels (actual transaction records) and negative labels (generated through random sampling). This approach helps the model learn to distinguish between positive and negative instances.

- **Loss Function**: Cross-entropy loss is commonly used as the loss function for binary classification tasks. It measures the dissimilarity between the predicted probability distribution and the actual labels.

#### Architecture Overview:

![xDeepFM Architecture](assets/xDeepFM.png)

### Advantages of xDeepFM:

- **Enhanced Feature Interactions**: The CIN layer allows the model to capture intricate relationships between features, leading to more accurate predictions.

- **Scalability**: xDeepFM is well-suited for large-scale datasets due to its efficient computation of feature interactions.

- **Flexibility**: The architecture can be adapted to various tasks in recommendation systems, advertising, and beyond.

By leveraging the capabilities of Extreme DeepFM, our forecasting model can effectively capture the complex dynamics of user-store interactions, thereby improving the accuracy of GMV predictions.

### Training Strategy: Rolling Window
•  Temporal Pattern Capture: Implement a rolling window strategy to effectively capture temporal patterns and address time autocorrelation.

•  Adaptability: Update the training dataset iteratively to maintain model relevance as the data distribution evolves over time.

•  Robust Predictions: Use historical data within each fixed-size window to inform the model, enhancing the accuracy and robustness of predictions.

•  Enhanced Features: Apply the rolling window method to calculate features akin to Recency, Frequency, and Monetary (RFM) metrics, like moving averages, providing deeper insights into the data trends.

Please note that the figure mentioned for the rolling window strategy is not displayed here, but it should be included in the final documentation where this strategy is presented.



## Result Reproducible

To reproduce the results, follow these steps:

1. **Prerequisite**: Ensure [Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) is installed.

2. **Environment Setup**: Run `make install` to set up the necessary environment.

3. **Training**: Execute `make train` to train the forecasting model.
