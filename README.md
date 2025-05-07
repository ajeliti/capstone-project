# SENTIMENT ANALYSIS OF SKINCARE PRODUCTS

![image](https://github.com/user-attachments/assets/55bdb4c5-a96c-4e00-b799-41a59c644857)

## Name of the Group: EYAI CYNDIKET
#### Members
1. Lionel Ajeliti
2. Kavata Musyoka
3. Tabby Mirara
4. AMos Kipkoech
5. Stanley Macharia

# Overview
From time immemorial, skin care has been a key factor for beauty. How well one takes care of  their skin can enhance or worsen their beauty.  Since skin care is vital for beauty, especially for women, companies have seized the opportunity to create skin care products.

However, sometimes the products don’t work well with certain skin types, or they don’t work as promised. These lead to customers complaining, or in other cases praising the products, if they help. These feedback is vital for the companies, since they help the companies know when the product is performing well or poorly.
Having a system/tool to look at customer feedback and analyze them is a key factor for the success of the company, since it helps to generate insights  and in turn improve products, and eventually  sales.

### PROJECT SUMMARY

This project uses Natural Language Processing (NLP) techniques to evaluate user-generated skincare product reviews, with the objective of identifying relevant sentiment patterns across distinct customer groups. The dataset comprises hundreds of evaluations, each providing subjective input on product efficacy, which is frequently linked to particular skin features such as tone and type.

The primary goal is to extract sentiment (positive, negative, or neutral) from free-text reviews and detect patterns associated with certain product features, skin types, and brands. The approach includes text preprocessing, tokenization, sentiment labeling, and model training with text classification-appropriate machine learning methods. Advanced visualizations and analysis are utilized to understand how sentiments change amongst demographic or skin-type groupings.

Unlike recommendation systems, which advise things, this NLP research focuses on understanding why certain products are evaluated positively or negatively, providing consumers and companies with deeper, data-driven understandings. By uncovering sentiment trends in large-scale textual data, our study helps to provide more transparent skincare experiences and data-driven product development.

### BUSINESS PROBLEM

In the beauty and skincare market, user evaluations are a valuable yet underutilized source of consumer information. These evaluations frequently include comprehensive personal experiences with products, emphasizing their impact on different skin kinds, tones, and conditions. However, due to the unstructured and subjective nature of this data, companies, researchers, and potential customers find it challenging to properly assess sentiment or establish trends across enormous amounts of input.

Most analytics now rely on star ratings or keyword mentions, which oversimplify user sentiment and fail to capture complex thoughts like mixed sentiments or conditional satisfaction (for example, "great for dry skin but irritating on sensitive areas"). This lack of granularity may lead to bad product development decisions, unproductive marketing tactics, and inadequate customer service.

This project deals with the demand for more understanding into skincare product evaluations by creating an NLP-powered sentiment analysis system. By using powerful natural language processing techniques to identify and evaluate user sentiment, the system hopes to derive significant patterns that represent real-world product success across a broad user base. The study will also look at links between sentiment and variables like skin tone, skin type, and brand, to get a better understanding of how various demographics react to skincare products.

### Objectives

Main Objective: To perform sentiment analysis on customer reviews on products to enhance customer satisfaction.

To use data visualizations tools to assess product categories and brand popularity to guide companies on future pricing
To assess price range across various products to improve affordability of products by customers.
To detect common keywords and phrases to highlight positive, neutral and negative reviews on products to understand customer nsatisfaction and dissatisfaction.
To provide actionable insights in order to improve customer satisfaction across various products, brands and categories.

### Stakeholders
Online stores that sell skin care products.
Companies that produce and sell skin care products.
Customers

### DATA UNDERSTANDING

The data was taken from kaggle. It contains information about beauty products from sephora online store.

The following are the key features for the dataset:

rating: The rating given by the author for the product on a scale of 1 to 5
is_recommended: Indicates if the author recommends the product or not (1-true, 0-false)
total_feedback_count: Total number of feedback (positive and negative ratings) left by users for the review
total_neg_feedback_count: The number of users who gave a negative rating for the review
total_pos_feedback_count: The number of users who gave a positive rating for the review
review_text: The main text of the review written by the author
review_title: The title of the review written by the author
skin_tone: Author's skin tone
skin_type: Author's skin type

### METRIC OF SUCCESS
A. Accuracy & Classification Metrics:
Accuracy: 88%

Measures overall correctness of sentiment predictions (positive, negative, neutral).

Precision: 85%

Reflects how many predicted positive sentiments are actually positive.

Recall: 83%

Measures how well the model identifies all actual positive sentiments.

F1 Score: 0.84

Harmonic mean of precision and recall, giving a single measure of model effectiveness.

B. Business & Engagement Metrics:
Sentiment Distribution Consistency: 95%

Checks if the sentiment classification follows expected distribution patterns across brands/products.

Top Brand Recognition Accuracy: 90%

Ensures the most positively reviewed brands align with actual high-performing brands in the dataset.

C. Coverage & Robustness:
Category Coverage: 100%

Ensures all product categories (e.g., moisturizers, cleansers, serums) are represented in sentiment classification.

N-gram Sentiment Generalization Score: 78%

Evaluates how well the model captures sentiment from varied linguistic patterns or less common phrasing.

Misclassification Rate on Ambiguous Reviews: < 10%

Assesses robustness by tracking errors in mixed or borderline sentiment texts.

### 1. Dataset Loading
We will load all the review datasets, check for null entries, merge them into one dataset and then drop unnecessary columns for Exploratory Data Analysis (EDA).
There are five datasets that were used. They were merged to form  a dataset that has 19 columns and 285,412 rows.

### 2. DATA CLEANING AND PRE-PROCESSING
We'll drop the unnecessary columns, impute and/or drop missing values.
Dropped irrelevant columns: Unnamed: 0, Unnamed: 0.1, helpfulness, and submission_time.

Missing value treatment:

Filled is_recommended with median.

Imputed skin_tone, eye_color, skin_type, and hair_color using the mode.

Dropped rows with missing review_text and review_title.

After cleaning, the dataset was reduced to 205,718 rows and 15 columns, with all missing values addressed.

#### Created a new column known as sentiment
Created a new Sentiment column based on the rating where:

Positive if rating > 3

Neutral if rating = 3

Negative if rating < 3

### DATA PRE-PROCESSING (on review_text)
Lowercased all text.

Removed noise: punctuation, digits, extra spaces, and expanded contractions.

Tokenized the cleaned text using NLTK.

Removed stopwords to focus on meaningful content.

These steps prepare the dataset for Exploratory Data Analysis (EDA) and further modeling tasks such as sentiment classification or recommendation.

3. ## EDA
   Exploratory Data Analysis (EDA) is performed to better understand the dataset before applying any models.
3.1 Univariate Analysis
   ### Sentiment Class Distribution
   ![image](https://github.com/user-attachments/assets/e7dc6edf-9c41-4761-b768-3b2b84c6d4ae)

 ### Top 20 brands
 ![image](https://github.com/user-attachments/assets/62f90257-8254-4178-a5e1-f372df61b261)

 ### Skin tone value counts
 ![image](https://github.com/user-attachments/assets/b24aba62-6686-4765-a0c3-a582e8f2ad71)

 ### Rating Distribution
 ![image](https://github.com/user-attachments/assets/5bf94c5c-b63a-4dd9-afcc-81dbe441f81b)

 ### Box plots to visualize outliers
 ![image](https://github.com/user-attachments/assets/23d1dd5b-71b1-4155-b125-43e5b786d3a0)

 ### 3.2 Bivariate Analysis
 ### Skin type vs Total feedback count
 ![image](https://github.com/user-attachments/assets/d8f70fe5-9c66-44e8-8661-62f1405d8a7e)

 ### Skin Tone vs Sentiment Distribution
 ![image](https://github.com/user-attachments/assets/d3ab9f04-7333-476f-a084-a24f48a6c167)

 ### Top 20 Most Expensive Brands
 ![image](https://github.com/user-attachments/assets/2e1cc159-973b-456e-8ade-d3565f4faa37)

 ### Top 20 Most Affordable Brands
 ![image](https://github.com/user-attachments/assets/b6dc80fa-3e1f-4fe9-8e6a-6fb603476627)

 ### 3.3 Multivariate Analysis
 
 ### Feedback counts
 ![image](https://github.com/user-attachments/assets/b6fce7a2-1ee7-422a-8b89-90cedec79882)

 ### Price Category Distribution
 ![image](https://github.com/user-attachments/assets/89ba015f-e132-47f9-aa32-4beae66758a4)

 ### Correlation
 ![image](https://github.com/user-attachments/assets/6024882e-ab1d-4c6c-aa95-b18a361b8d98)

 ### 4. Feature Engineering
#### Term Frequency–Inverse Document Frequency
To convert our text into a format that machine learning models can process, we transform the cleaned review text into numerical features through vectorization.

### WordCloud for most frequent words
![image](https://github.com/user-attachments/assets/a67de0f0-fac5-414b-8ec1-e5113560cb00)

We create a new column sentiment in order to classify our ratings into positive, negative and neutral.

Positive sentiments dominate the data, as seen from the previous graph of rating distribution. We'll now create new dataframes according to sentiments so that we can use them to create word clouds for those sentiments.

We'll use review_title column to explore the word clouds for those sentiments.

### WordCloud for positive sentiments

![image](https://github.com/user-attachments/assets/e9ee5ffd-9505-40aa-967d-3d1e7d0e0a71)

### WordCloud for negative sentiments
Explore negative sentiments

![image](https://github.com/user-attachments/assets/2bff7cee-5794-46fb-96a5-865acc9db3ed)

### WordCloud for neutral sentiment
Explore neutral sentiments
 
![image](https://github.com/user-attachments/assets/547a45f9-70cb-481c-bbf8-035882bbb7bb)

### Bigram Analysis
We'll now perform a bigram analysis to see which words appear together frequently

![image](https://github.com/user-attachments/assets/55969ae9-f926-4893-814f-02c05fcd9911)

### 5. MODELLING
We will start with defining our target and features, train and test split, then we balance the training set.

### 5.1 Baseline Model

We trained a logistic regression model as our baseline. The model achieved:

Accuracy: 86%

F1-score (positive class): 0.91

F1-score (negative class): 0.56

This shows strong performance on positive predictions but highlights room for improvement on negative ones.

### Confusion Matrix:
We visualized the prediction results with a confusion matrix to better understand how well the model distinguishes between recommended and not recommended products.


![image](https://github.com/user-attachments/assets/effc4fff-80d1-4ef0-b8a0-31238c84070f)

### Feature Importance
We analyzed the model’s feature importance to see which words influenced the prediction the most. Words with high positive scores were strong indicators of recommended products (like “amazing” or “love”), while words with high negative scores were linked to negative reviews (like “dry” or “disappointed”). This helps us understand which terms drive customer sentiment.

### Cross Validation
We used cross-validation to test how well our model performs on different subsets of the data. By splitting the data into 5 parts and rotating the training/testing process, we got a more reliable estimate of model performance. The average F1 score from cross-validation shows the model’s ability to balance precision and recall across multiple runs.

### Tune Hyperparameters
We tuned the model’s hyperparameters to improve its performance. Specifically, we tested different values of C (which controls regularization strength) using the 'l2' penalty. This helps the model generalize better and avoid overfitting by finding the best settings through a grid search.

### Random Forest
The model correctly identified recommended products most of the time.

It did a decent job at catching not recommended ones too, though with more mistakes than for the recommended.

Overall, it achieved 84% accuracy, showing good performance with some room for improvement in handling negative cases

### Confusion Matrix

![image](https://github.com/user-attachments/assets/1ce05e8f-a112-48a1-8ef8-ad9562a295b2)

### Naive Bayes
The model did well in identifying recommended products, showing high precision and recall.

It struggled more with correctly detecting not recommended products, misclassifying many of them.

Despite this, the model reached 83% overall accuracy, making it a strong performer for positive cases but less reliable for negatives.

### Confusion Matrix

![image](https://github.com/user-attachments/assets/7a85539d-246f-4e33-839a-f182b06ce263)

### XGBoost

Model Performance: The model achieved 87% accuracy on the test set, with strong precision and recall for the recommended products (1.0 class). It performed slightly worse on the not recommended products (0.0 class), showing a precision of 0.48 and recall of 0.70.

### Confusion Matrix

![image](https://github.com/user-attachments/assets/65e37ccc-bf8e-454b-9fea-5c63a915de4e)

### 6. Model Evaluation
### ROC Curve for Model Comparison

![image](https://github.com/user-attachments/assets/455580f9-a1bf-47f8-bc49-96e95dbbc451)

### Logistic Regression
Logistic regression is a key model for multiclass classification problems. Because of its simplicity, quickness, and interpretability, it serves as an excellent starting point for initial examination.

It helps to develop an initial baseline for detecting product recommendations based on TF-IDF features. The model was simple to interpret. Its coefficients revealed which features were more influential. Cross-validation was used to evaluate the model and tweak it for improved F1 performance.

With an AUC Score of 0.90 this implies that the model did well in discriminating between recommended and non-recommended items, providing a fair evaluation between sensitivity and specificity.

### Random Forest Classifier
Random Forests are ensemble models which combine numerous decision trees to increase accuracy while minimizing overfitting. They perform well with high-dimensional datasets such as TF-IDF vectors and can detect complicated feature relationships.

The model outperformed logistic regression by learning nonlinear connections. Its efficacy was further enhanced by hyperparameter optimization.

With an AUC Score of 0.92 which was the highest score across all models, indicating that Random Forest performed the best in terms of balancing true and false positives. It displayed remarkable accuracy in recognizing recommended products.

### Naive Bayes
Naive Bayes is widely used in text categorization due to its simplicity and performance with high-dimensional inputs such as word frequencies or TF-IDF scores.

Naive Bayes is simple to train and fast, it assumes feature independence, which restricts its capacity to grasp correlations between words. It is best used as a lightweight benchmark to compare with more complicated models.

It was the lowest of the models, with the least AUC Score showing that it struggled to strike an adequate balance between recall and specificity. It was the least appropriate model in this circumstance.

### XGBOOST
XGBoost is a high-performance gradient boosting technique popular for its speed, regularization, and predictive capability. It excels at managing unbalanced datasets and nonlinear patterns.

XGBoost combined competitive performance with more flexibility. The model's tweaking helped increase accuracy and recall, but it did not outperform Random Forest in AUC. Nonetheless, it provided a robust alternative with outstanding learning ability.

Checking on the AUC Score of 0.90 the model has the same AUC score as the Logistic regression model.

Random Forest outperformed XGBOOST.

### 7. CONCLUSION AND RECOMMENDATIONS

#### 7.1 Conclusions

From the above, the Random Forest model has the highest AUC of 0.92 thus has the best classification performance compared to the other models as it provides the most accurate trade-off between sensitivity (recall) and specificity.

The logistic regression model has a good performance of 0.90 that indicates its capable of distinguishing between the positive and negative classes. It provides a strong balance between precision and recall.

Naive Bayes has the lowest AUC of 0.77

#### 7.2 Recommendations

We Recommend-
1. Personalized Product Displays for online stores
Use skin tone and sentiment analytics to dynamically surface goods in your online and mobile catalogs. For instance, if consumers with darker skin tones have a greater favorable attitude for a specific moisturizer, it should be promoted more prominently.

2. DEI(Diversity, Equity, and Inclusion) Transparency for companies
Publish quarterly "Skin-Tone Sentiment Scores" with CSR(Corporate Social Responsibility) reports to demonstrate commitment and development over time.

3. Promotional Targeting for online stores
Schedule flash sales or bundle offers based on lagging sentiments on products. For example, give a "Buy One, Get One" on goods with neutral feedback to encourage a test.

4. Learning Tutorials/material for customers on how to use specific products
Generate educational materials explaining the best products that most customers preffer and the different price categories based on customer's budget.

#### 7.3 Next Steps

1. Personalized Recommendation Widget.
Create an on-site "What Works for You" survey where buyers may select their skin tone and obtain a list of top-rated products based on sentiments, including user quotations.

2. Creating a real-time sentiment insight dashboard for online stores.
This will generate reports based on real-time sentiments analysis, product monitoring, marketing insights and proactive customer service action.



















