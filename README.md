# Travelers-Modelling-Competition
Performed classification algorithm on insurance dataset to predict the occurrence of fraudulent claims given a training set and testing set. 
Team members: Kah Meng, Audrey, Lukas, and Li. 
Models: Tree-based, logistic, Boosting, KNN, Naives Bayes, and SVM. 
Remarks: the logistic regression is default with l2 (ridge) with no shrinkage penalty (default C=1), should apply MultinomialNB instead of GaussianNB as they are more categorical predictors than continuous predictors, can also bin continuous predictors to appropriately applied for MultinomialNB.
