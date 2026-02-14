
1. Objective
    1. The objective of this project is to accurately forecast electricity consumption to ensure a stable and efficient energy supply. By leveraging temporal data and meteorological features, this machine learning model is designed to discover power demand (electricity usage) forecasting algorithms that can be actively deployed in industrial settings. This project serves to explore and validate the practical applicability of machine learning regression model in real-world energy management
2. Data
    1. This machine learning model leverages temporal and meteorological factors that influence energy demand. Non-numerical and categorical data types were conversed into numerical data types, and because weather feature have vastly difference scales, StandardScaler was applied for preprocessing. The data was split into training and testing sets to evaluate the model’s forecasting accuracy on unseen future data.
3. Used Model
    1. Before testing out different regression models, K-Means Clustering was applied to discover groupings within the provided data. Elbow Method was utilized to determine the optimal number of clusters. 
    2. Then, to evaluate a set of regression algorithms, PyCaret was leveraged and allowed for comparison of different regression model performances. 
    3. Based on the PyCaret results, Decision Tree, K-Nearest Neighbor, and Huber regressors were chosen. With the Decision Tree Regressor, key hyper parameters were max_depth for preventing tree’s overgrowing and min_samples_split to control the minimum number of samples required to split in a node. In KNN Regressor, key hyper parameters were n_neighbors and weights to determinate number of closest data points to consider. With the Huber Regressor, alpha and epsilon was used. To maximize forecasting accuracy for electricity consumption, a Stacking Regressor was implemented with the tree chosen models.
4. Outcome
    1. 
