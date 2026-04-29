# Question and Background Information
Credit card fraud is a major and growing issue in today’s financial system. According to the Nilson Report, global card fraud losses exceeded $33 billion in 2024, affecting card issuers, merchants, and payment processors. Financial institutions must detect fraudulent transactions quickly — ideally in real-time — to minimize losses and protect customers.

However, fraud detection is inherently challenging because fraudulent transactions are extremely rare. In most datasets, they account for less than 1% of all transactions. This creates a severe class imbalance problem in machine learning. For example, a model that simply predicts every transaction as “legitimate” could achieve over 99% accuracy while failing to detect any fraud at all. As a result, traditional accuracy metrics are misleading, and many models become biased toward the majority (legitimate) class.

This leads to the central question: Can machine learning models reliably detect fraudulent transactions despite extreme class imbalance? If so, which techniques are most effective for addressing this issue? 

To explore this, we use the Credit Card Fraud Detection dataset from Kaggle. It contains 284,807 transactions made by European cardholders in September 2013, of which only 492 (0.173%) are fraudulent. Most features (V1–V28) are anonymized using PCA for confidentiality, while “Time,” “Amount,” and “Class” (0 = legitimate, 1 = fraud) remain in their original form. Taken together, this dataset is challenging yet also representative of real-world fraud detection problems. 

In the notebook titled "cc_fraud.ipynb" the reader can find some exploratory data analysis, as well as 2 Logistic Regression Models: Logistic Regression with Balanced Class Weights and Logistic Regression with Oversampling. In the notebook entitled "ExtraTrees_and_RandomForest.ipynb", the reader can find our Decision Tree Models.

# Conclusion
Credit card fraud is a very rare occurrence for the anonymous credit card company illustrated in our dataset, but the people who are affected by it undergo serious financial harm.  The success of a machine learning model to catch fraudulent transactions would contribute to this company's credibility.  Therefore, we would employ an ExtraTreesClassifier (Model 5) model to identify fraudulent transactions if we worked for this company.  We came to this conclusion based primarily on the metrics for false negatives and positives but also considered the Precision-Recall AUC scores from the additional Logistic Regression and RandomForestClassifier models that we made.   

Across all logistic regression models, we found that performance is highly sensitive to how class imbalance is handled and how decision thresholds are set. While default models achieved high recall, they produced an unacceptable number of false positives, making them impractical for real-world use. Introducing class weighting and especially tuning the classification threshold significantly improved the balance between precision and recall, with Model 1b emerging as the strongest logistic regression approach. However, even the best logistic regression model remained limited in its ability to fully separate fraudulent from legitimate transactions, motivating the need to explore more flexible, nonlinear methods.

The ExtraTreesClassifier model, which we implemented because of its ability to balance class weights, delivered extremely promising results with a PR-AUC of 0.8758 and an ROC-AUC of 0.9528.  This model caught 83 out of 98 real fraudulent cases and only had 8 false alarms out of 56,864 legitimate transactions logged in the dataset. These metrics indicate that in comparison to Logistic Regression, an ExtraTreesClassifier is more successful at handling imbalanced data without risking data leakage.  The false positives and negatives represented transactions that were indistinguishable from true positives and negatives, as their probabilities of being their true labels were too comparable to that of true predictions.  Therefore, the imperfections of this model were unavoidable based on the information provided by the anonymous credit card company.  We tested this conclusion by creating a RandomForestClassifier for our final model. 


We compared a RandomForestClassifier model to our previous models to determine if finding optimal splits in tree-based modeling produced better results than random splits completed by the ExtraTreesClassifier.  When implemented, this model proved to catch slightly more fraudulent cases.  Despite this improvement, the model caught one more false alarm and had a slightly lower PR-AUC than the ExtraTreesClassifier model.  This difference was enough to confirm that the ExtraTreesClassifier model performed the best given available data.   Therefore, it is possible to answer our initial question and use a machine learning model to correctly identify fraudulent transactions despite their scarcity, and the most optimal model to do so is an ExtraTreesClassifier.   



# Team Contribution
Claire: Writing background information and question, doing the Exploratory Data Analysis, creating, deploying, and evaluating the logistic regression models

Macy: Identifying dataset, ExtraTreesClassifier and RandomForestClassifier models with corresponding evaluations, and conclusion



# Link to dataset (origin of data): 
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download
