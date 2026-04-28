# Question and Background Information



# Conclusion
Credit card fraud is a very rare occurrence for the anonymous credit card company illustrated in our dataset, but the people who are affected by it undergo serious financial harm.  The success of a machine learning model to catch fraudulent transactions would contribute to this company's credibility.  Therefore, we would employ an ExtraTreesClassifier (Model 5) model to identify fraudulent transactions if we worked for this company.  We came to this conclusion based primarily on the metrics for false negatives and positives but also considered the Precision-Recall AUC scores from the additional Logistic Regression and RandomForestClassifier models that we made.   


Oversampling information for the final Logistic Regression model, which had a PR-AUC of 0.99, was helpful for ensuring that the severe imbalance in favor of legitimate transactions did not impact its ability to identify fraudulent transactions.  However, the model failed to correctly classify 3,541 fraudulent transactions and 2042 legitimate transactions, most of which were synthetically generated in the test set to minimize class bias.  These are symptoms of data leakage.  While the use of Logistic Regression was insightful for our introductory models, these results prompted us to compare Logistic Regression to the usefulness of an ExtraTreesClassifier. 


The ExtraTreesClassifier model, which we implemented because of its ability to balance class weights, delivered extremely promising results with a PR-AUC of 0.8758 and an ROC-AUC of 0.9528.  This model caught 83 out of 98 real fraudulent cases and only had 8 false alarms out of 56,864 legitimate transactions logged in the dataset. These metrics indicate that in comparison to Logistic Regression, an ExtraTreesClassifier is more successful at handling imbalanced data without risking data leakage.  The false positives and negatives represented transactions that were indistinguishable from true positives and negatives, as their probabilities of being their true labels were too comparable to that of true predictions.  Therefore, the imperfections of this model were unavoidable based on the information provided by the anonymous credit card company.  We tested this conclusion by creating a RandomForestClassifier for our final model. 


We compared a RandomForestClassifier model to our previous models to determine if finding optimal splits in tree-based modeling produced better results than random splits completed by the ExtraTreesClassifier.  When implemented, this model proved to catch slightly more fraudulent cases.  Despite this improvement, the model caught one more false alarm and had a slightly lower PR-AUC than the ExtraTreesClassifier model.  This difference was enough to confirm that the ExtraTreesClassifier model performed the best given available data.   Therefore, it is possible to answer our initial question and use a machine learning model to correctly identify fraudulent transactions despite their scarcity, and the most optimal model to do so is an ExtraTreesClassifier.   



# Team Contribution
Claire: 

Macy: Identifying dataset, ExtraTreesClassifier and RandomForestClassifier models with corresponding evaluations, and conclusion



# Link to dataset (origin of data): 
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download
