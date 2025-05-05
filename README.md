# SENTIMENT ANALYSIS OF SKINCARE PRODUCTS

![image](https://github.com/user-attachments/assets/55bdb4c5-a96c-4e00-b799-41a59c644857)

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
Stakeholders
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

![image](https://github.com/user-attachments/assets/a36254b8-d7f5-4078-b21b-f03cd14c53b5)


