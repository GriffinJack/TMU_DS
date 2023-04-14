#Create a function that takes in the base learner, training data, traing labels and unlabeled data
#Inside the function, implement 3 different uncertainty measures.
#Least Confident, Margin, and Entropy. 

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def uncertainty_sampling(baseLearner, X_train, y_train, X_remained):

    df = X_remained

    #train the base learner on the training data 
    baseLearner.fit(X_train, y_train)

    #predict labels and class probability for all unlabeled data
    pred_labels = baseLearner.predict(X_remained)
    pred_prob = pd.DataFrame(baseLearner.predict_proba(X_remained))
    pred_prob.columns = ['prob_c1', 'prob_c2', 'prob_c3', 'prob_c4', 'prob_c5']
    #Add new column with predicted labels
    df['pred_label'] = pred_labels

    #Add columns for class probability
    df = pd.concat([df,pred_prob.reindex(df.index)], axis=1)
    #add columns for uncertainty sampling
    df['LC'] = leastConfident(pred_prob)
    df['Margin'] = margin(pred_prob)
    df['Entropy'] = entropy(pred_prob)

    return df


#create a function to handle least confident
#least confident is the diference between the highest confident and 1
def leastConfident(probs):
    LC = []
    for i in range(probs.shape[0]):
        LC.append(round(1-probs.loc[i].max(),2))
    return LC

#create a function to handle margin
#margin is the difference between the two highest probs
def margin(probs):
    Margin = []
    for i in range(probs.shape[0]):
        first = probs.loc[i].sort_values(ascending = False)[0]
        second = probs.loc[i].sort_values(ascending = False)[1]
        Margin.append(first-second)
    return Margin

#create a function to handle entropy
def entropy(probs):
    Entropy = []
    print(probs.loc[0].shape[0])
    for i in range(probs.shape[0]):
        Entropy.append(0-np.sum(probs.loc[i] * np.log(probs.loc[i]))/ np.log(probs.loc[i].shape[0]))
    return Entropy


    


df_labeled = pd.read_csv('AL_data_labeled.csv')
df_unlabeled = pd.read_csv('AL_data_unlabeled.csv')

y_col = 'label'
x_cols = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5', 'feat6', 'feat7']
y = df_labeled[y_col]
X = df_labeled[x_cols]

#uncertainty_sampling(RandomForestClassifier(random_state=42), X, y, df_unlabeled)

print(uncertainty_sampling(RandomForestClassifier(random_state=42), X, y, df_unlabeled))