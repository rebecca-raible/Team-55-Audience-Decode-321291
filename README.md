# Audience Decode: Behavioural Patterns in Streaming

Team Members: Rebecca Raible, Eleonora Ciufoli, Yara Messina, Mila Kirova

## 1. INTRODUCTION
This project analyzes the viewer_interactions.db dataset to uncover behavioural patterns within a large-scale streaming platform, moving beyond simple rating prediction to understand the underlying drivers of viewer engagement. 

Through dual-layer segmentation, we analyzed both the audience and content library through unsupervised methods like clustering. Allowing us to identified distinct behavioural personas, revealing that viewing patterns are not uniform; instead, users belong to specific segments that dictate how they interact with content. By analyzing how these user archetypes interact with different forms of content we were able to provide strategic insights for the platform to adopt in order to grow and provide targeted recommendations. 

Through a rigorous comparison of Linear Regression, Decision Trees, Random Forests, and Artificial Neural Networks (ANN), we evaluate whether these complex interactions between audience and content are better captured by interpretable linear trends or sophisticated non-linear dynamics.

## 2. METHODS
### 2.1 Data Overview
To prepare the raw interaction data for modelling, we implemented a rigorous pre-processing pipeline. The dataset was stored in a relational database consisting of multiple tables capturing different aspects of the system:
1.	`viewer_ratings`: individual ratings given by users to movies, including the rating value and the date of interaction.
2.	`user_statistics`: aggregated statistics describing user behaviour (e.g. average rating given by a user, number of ratings).
3.	`movie_statistics`: aggregated statistics describing movie popularity and reception (e.g. average movie rating, number of users who rated the movie).
4.	`movies`: metadata about movies, including movie_id, title, and year_of_release.


### 2.2 Exploratory Data Analysis (EDA)

### 2.3 Data Processing & Cleaning

### 2.4 Feature Engineering
Feature engineering focused on transforming raw, interaction-level data into interpretable behavioural representations at the user and content level. Both for our clustering and regression models, we engineered aggregated behavioural features including: 
1.	User-level statistics (e.g. average rating given by the user, rating variability)
2.	Movie-level statistics (e.g. average rating received by the movie, number of unique users)
3.	Optional temporal metadata (year_of_release)
4.	User-Content matrices (e.g preference and exposure matrices)
Identifiers such as `customer_id`, `movie_id`, and textual fields are excluded from the feature matrix, as they do not convey ordinal or numerical meaning useful for our prediction and could introduce noise. As well as redundant features such as `min_rating`/`max_rating` were dropped.


### 2.5 Modelling and Analysis Techniques
The modelling approach focuses on descriptive and explanatory techniques to uncover behavioural patterns in viewer engagement and content consumption, this was done through:

•	 **Clustering for audience and content segmentation**: clustering methods such as K-Means++, HDBSCAN and an autoencoder were used to identify groups with similar behavioural characteristics. User clustering captures differences in engagement intensity, rating strictness and activity duration whereas content clustering distinguishes items by popularity and audience reception patterns

•	 **Preference and exposure analysis**: aggregated user-content matrices were constructed to analyze how different audience segments interact with different content groups

•	 **Temporal analysis of behavioural change**: to study how preferences evolve from the platform’s establishment period to their high-growth period, interactions were partitioned into temporal segments and the difference in preferences across these periods were analyzed 

•	 **Individual rating prediction**: supervised learning techniques were trained to predict individual user ratings based on engineered behavioural and content features:

  o	 *Linear Regression (Baseline)*: provides immediate interpretability through coefficients, allowing us to quantify exactly how much a feature impacts the final rating
  
  o	 *Artificial Neural Network (Prediction)*: Multi-Layer Perceptron (MLP) using the Keras Sequential API with Dense layers and ReLU activation functions, utilized the Adam optimizer and Mean Squared Error loss
  
  o	 *Decision Tree Regression*: used due to its ability to capture non-linear relationships and its interpretability
  
  o	 *Random Forest Regressor*: employed as an ensemble extension of Decision Trees, aiming to reduce variance and improve generalization
The models were trained using aggregated user-level and movie-level numerical features derived from the merged dataset. Model hyperparameters such as tree depth, minimum number of samples per leaf, and number of estimators were selected to balance model complexity and overfitting.



