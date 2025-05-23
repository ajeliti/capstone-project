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

-To use data visualizations tools to assess product categories and brand popularity to guide companies on future pricing

-To assess price range across various products to improve affordability of products by customers.

-To detect common keywords and phrases to highlight positive, neutral and negative reviews on products to understand customer nsatisfaction and dissatisfaction.

-To build a model that recommends products based on skin type, skin tone and price.

### Stakeholders
-Online stores that sell skin care products.

-Companies that produce and sell skin care products.

-Customers

### DATA UNDERSTANDING

The data was taken from kaggle. It contains information about beauty products from sephora online store.

The following are the key features for the dataset:

-rating: The rating given by the author for the product on a scale of 1 to 5

-is_recommended: Indicates if the author recommends the product or not (1-true, 0-false)

-total_feedback_count: Total number of feedback (positive and negative ratings) left by users for the review

-total_neg_feedback_count: The number of users who gave a negative rating for the review

-total_pos_feedback_count: The number of users who gave a positive rating for the review

-review_text: The main text of the review written by the author

-review_title: The title of the review written by the author

-skin_tone: Author's skin tone

-skin_type: Author's skin type

### METRIC OF SUCCESS

#### A. Accuracy & Classification Metrics:
Accuracy: 88%

Measures overall correctness of sentiment predictions (positive, negative, neutral).

Precision: 85%

Reflects how many predicted positive sentiments are actually positive.

Recall: 83%

Measures how well the model identifies all actual positive sentiments.

F1 Score: 0.84

Harmonic mean of precision and recall, giving a single measure of model effectiveness.

#### B. Business & Engagement Metrics:
Sentiment Distribution Consistency: 95%

Checks if the sentiment classification follows expected distribution patterns across brands/products.

Top Brand Recognition Accuracy: 90%

Ensures the most positively reviewed brands align with actual high-performing brands in the dataset.

#### C. Coverage & Robustness:
Category Coverage: 100%

Ensures all product categories (e.g., moisturizers, cleansers, serums) are represented in sentiment classification.

N-gram Sentiment Generalization Score: 78%

Evaluates how well the model captures sentiment from varied linguistic patterns or less common phrasing.

Misclassification Rate on Ambiguous Reviews: < 10%

Assesses robustness by tracking errors in mixed or borderline sentiment texts.

### 2. DATA CLEANING AND PRE-PROCESSING
We'll drop the unnecessary columns, impute and/or drop missing values.

Lowercased all text.

Removed noise: punctuation, digits, extra spaces, and expanded contractions.

Tokenized the cleaned text using NLTK.

Removed stopwords to focus on meaningful content.

These steps prepare the dataset for Exploratory Data Analysis (EDA) and further modeling tasks such as sentiment classification or recommendation.

3. ## EDA
   Exploratory Data Analysis (EDA) is performed to better understand the dataset before applying any models.

   
### Top 20 brands
 ![image](https://github.com/user-attachments/assets/62f90257-8254-4178-a5e1-f372df61b261)

CLINIQUE was the most popular in the dataset and SEPHORA COLLECTION was the least.
 

### Rating Distribution

![image](https://github.com/user-attachments/assets/c02a78fe-9b6f-4283-8ea8-d761b4ca5c1f)

Most products are highly rated
 
 
 ### Skin type vs Total feedback count
 
 ![image](https://github.com/user-attachments/assets/9e38552d-173a-46d5-bf3e-64755a7d1dd6)

 Majority of the population had combination and dry skin types

 

 ### Price Category Distribution

 ![image](https://github.com/user-attachments/assets/e7e27026-f8b7-4011-8207-a115e9a2f8a8)


Most products range between 20 50 dollars. This shows a big percentage of products are affordable
 
### WordCloud for positive sentiments

![image](https://github.com/user-attachments/assets/e9ee5ffd-9505-40aa-967d-3d1e7d0e0a71)

Words like “Amazing,” “Love,” “Great,” and “Good” stand out.

### WordCloud for negative sentiments

![image](https://github.com/user-attachments/assets/2bff7cee-5794-46fb-96a5-865acc9db3ed)

Words like “worth,” “skin,” “sensitive,” “drying,” and “money” .


### 5. MODELLING
We will start with defining our target and features, train and test split, then we balance the training set.


We trained a logistic regression model as our baseline. The model achieved:

How all the models performed.

![Capture100](https://github.com/user-attachments/assets/5ca1da8d-7856-4472-a197-7e4454536784)



### 6. Model Evaluation
### ROC Curve for Model Comparison

![image](https://github.com/user-attachments/assets/73bc3236-1cf0-4ccc-a795-a5b3b743f26c)

### Logistic Regression
Logistic regression is a key model for multiclass classification problems. Because of its simplicity, quickness, and interpretability, it serves as an excellent starting point for initial examination.

It helps to develop an initial baseline for detecting product recommendations based on TF-IDF features. The model was simple to interpret. Its coefficients revealed which features were more influential. Cross-validation was used to evaluate the model and tweak it for improved F1 performance.

With an AUC Score of 0.90 this implies that the model did well in discriminating between recommended and non-recommended items, providing a fair evaluation between sensitivity and specificity.

### Random Forest Classifier
Random Forests are ensemble models which combine numerous decision trees to increase accuracy while minimizing overfitting. They perform well with high-dimensional datasets such as TF-IDF vectors and can detect complicated feature relationships.

The model outperformed logistic regression by learning nonlinear connections. Its efficacy was further enhanced by hyperparameter optimization.

With an AUC Score of 0.92 which was the highest score across all models, indicating that Random Forest performed the best in terms of balancing true and false positives. It displayed remarkable accuracy in recognizing recommended products.

### Linear SVC
Linear SVC is a type of support vector machine implemented for classification tasks when there needs a separation between the features and class labels by means of a linear decision boundary. Linear SVC performs well in high dimensional data, robust to overfitting, and good for classification.

93% of the time, the model gives correct predictions. Although it is skewed a little bit by dominance of class 1, it has a pretty good macro average F1-score of 87%
### XGBOOST
XGBoost is a high-performance gradient boosting technique popular for its speed, regularization, and predictive capability. It excels at managing unbalanced datasets and nonlinear patterns.

XGBoost combined competitive performance with more flexibility. The model's tweaking helped increase accuracy and recall, but it did not outperform Random Forest in AUC. Nonetheless, it provided a robust alternative with outstanding learning ability.

Checking on the AUC Score of 0.90 the model has the same AUC score as the Logistic regression model.

Random Forest outperformed XGBOOST.

### 7. CONCLUSION AND RECOMMENDATIONS

#### 7.1 Conclusions

We picked XGBOOST because it was the best performing model for its excellent classification power as it provides the most accurate trade-off between sensitivity (recall) and specificity.

#### 7.2 Recommendations
We recommend:

-Personalized Product Displays for online stores.

-DEI(Diversity, Equity, and Inclusion) Transparency for companies.

-Promotional Targeting for online stores.

-Generate educational content for customers on the range and pricing of products

-Product use


#### 7.3 Next Steps

-Creating a real-time sentiment insight dashboard for online stores.
This will generate reports based on real-time sentiments analysis, product monitoring, marketing insights and proactive customer service action.

-Collect data to tailor it for the Kenyan market.



















