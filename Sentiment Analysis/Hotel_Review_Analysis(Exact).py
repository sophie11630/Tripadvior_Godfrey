#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 23:05:21 2020

@author: sophie
"""

#%%%%#Import libraries
import os
import pandas                        as pd
import numpy                         as np
import matplotlib.pyplot             as plt
from wordcloud                       import WordCloud, STOPWORDS 
from sklearn                         import tree
from sklearn                         import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model            import LinearRegression
from sklearn.neighbors               import KNeighborsClassifier
from sklearn.naive_bayes             import GaussianNB

pd.set_option('display.max_rows',     20)
pd.set_option('display.max_columns',  20)
pd.set_option('display.width',       800)
pd.set_option('display.max_colwidth', 20)

np.random.seed(1)
os.chdir('/Users/sophie/Desktop/BUS 256 - Marketing Analytics/Final Project/Godfrey Review')


#%%%% Read data
dta = pd.read_csv("Godfrey_review_data.csv")


#%%%% Split the data
dta['ML_group']   = np.random.randint(100,size = dta.shape[0])
dta['N_stars']    = dta.rating
dta               = dta.sort_values(by='ML_group').reset_index()
inx_train         = dta.ML_group<80                     
inx_valid         = (dta.ML_group>=80)&(dta.ML_group<90)
inx_test          = (dta.ML_group>=90)


#%%%% Putting structure in the text
corpus          = dta.review_text.to_list()
ngram_range     = (1,1)
max_df          = 0.85 
min_df          = 0.01
vectorizer      = CountVectorizer(lowercase   = True,
                                  ngram_range = ngram_range,
                                  max_df      = max_df     ,
                                  min_df      = min_df     );
                                  
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())
print(X.toarray())

#%%%% create a wordcloud graph
#quated from https://www.geeksforgeeks.org/generating-word-cloud-python/

words = dta[['review_text','rating']]
words = words[pd.notnull(words['review_text'])]
words = words[words['review_text']!=0]
words = words.sort_values('rating', ascending=[0])
top10 = words.head(10)

comment_words = '' 
stopwords = set(STOPWORDS)
stopwords.update(["will", "stay", "boston", "hotel", "room", "back", "go"])
  
# iterate through the csv file 
for val in top10.review_text: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
                         
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


#%%%% TVT-splitting
Y_train = dta.N_stars[inx_train].to_list()
Y_valid = dta.N_stars[inx_valid].to_list()
Y_test  = dta.N_stars[inx_test].to_list()

X_train = X[np.where(inx_train)[0],:]
X_valid = X[np.where(inx_valid)[0],:]
X_test  = X[np.where(inx_test) [0],:]

right_to_total_dict = {}


#%%%% Linear Regression
model  = LinearRegression()
clf1 = model.fit(X_train, Y_train)

dta['N_star_hat_reg'] = np.concatenate(
        [
                clf1.predict(X_train),
                clf1.predict(X_valid),
                clf1.predict(X_test )
        ]
        ).round().astype(int)

dta.loc[dta['N_star_hat_reg']>5,'N_star_hat_reg'] = 5
dta.loc[dta['N_star_hat_reg']<1,'N_star_hat_reg'] = 1

# confusion matrix for linear regression
conf_matrix1 = np.zeros([5,5])
for i in range(5):
    for j in range(5):
        conf_matrix1[i,j] = np.sum((dta[inx_test].N_stars == i+1)*(dta[inx_test].N_star_hat_reg == j+1))

print(conf_matrix1)

# print the probability of right(almost right) prediction
right = conf_matrix1.diagonal(offset = 0).sum() 
total = conf_matrix1.sum()
right_to_total = round(right/total,4)
print(right_to_total)

right_to_total_dict["Linear regression"] = right_to_total


#%%%% Naive Bayes Classification
clf2 = GaussianNB().fit(X_train.toarray(), Y_train)
dta['N_star_hat_NB']             = np.concatenate(
        [
                clf2.predict(X_train.toarray()),
                clf2.predict(X_valid.toarray()),
                clf2.predict(X_test.toarray( ))
        ]).round().astype(int)

dta.loc[dta['N_star_hat_NB']>5,'N_star_hat_NB'] = 5
dta.loc[dta['N_star_hat_NB']<1,'N_star_hat_NB'] = 1

# confusion matrix for  Naive Bayes
conf_matrix2 = np.zeros([5,5])
for i in range(5):
    for j in range(5):
        conf_matrix2[i,j] = np.sum((dta[inx_test].N_stars == i+1)*(dta[inx_test].N_star_hat_NB == j+1))    

print(conf_matrix2)

#print the probability of right(almost right) prediction
right = conf_matrix2.diagonal(offset = 0).sum()
total = conf_matrix2.sum()
right_to_total = round(right/total,4)
print(right_to_total)

right_to_total_dict["Naive Bayes"] = right_to_total


#%%%% trees
criterion_chosen     = ['entropy','gini']
random_state         = 96
max_depth            = 10
criterion_best       = {}
results_tree         = []
#select criterion and depth -- if the results are same, I chose the smallest depth and entropy
for criterion in criterion_chosen:
    k_dict_tree          = {}
    results_list         = []
    for depth in range(2,max_depth):
        clf3    = tree.DecisionTreeClassifier(
                criterion    = criterion, 
                max_depth    = depth,
                random_state = 96).fit(X_train.toarray(), Y_train)
    
        results_list.append(
            np.concatenate(
                    [
                            clf3.predict(X_train.toarray()),
                            clf3.predict(X_valid.toarray()),
                            clf3.predict(X_test.toarray())
                    ]).round().astype(int)
            )
        
        dta_results_tree              = pd.DataFrame(results_list).transpose()
        dta_results_tree.loc[dta_results_tree[depth-2]>5,depth-2] = 5
        dta_results_tree.loc[dta_results_tree[depth-2]<1,depth-2] = 1
        dta_results_tree['inx_train'] = inx_train.to_list()
        dta_results_tree['inx_valid'] = inx_valid.to_list()
        dta_results_tree['inx_test']  = inx_test.to_list()
        dta_results_tree["N_stars"]   = dta.N_stars.copy()
    
        
        # use validating data for choosing hyperparameter
        conf_matrix3 = np.zeros([5,5])
        for i in range(5):
            for j in range(5):
                conf_matrix3[i,j] = np.sum((dta_results_tree[dta_results_tree.inx_valid].N_stars == i+1) * (dta_results_tree[dta_results_tree.inx_valid][depth-2] == j+1))
                
        #print the probability of right(almost right) prediction
        right = conf_matrix3.diagonal(offset = 0).sum()
        total = conf_matrix3.sum()
        
        
        #add k and its right-to-total ratio to the dictionary
        k_dict_tree[depth] = round(right/total,4)
        
    # print dictionary and find the smallest best depth
    print(k_dict_tree)
    
    ratio_list = list(k_dict_tree.values())
    max_index = ratio_list.index(max(ratio_list))
    better_depth = list(k_dict_tree.keys())[max_index]
    
    criterion_best[criterion] = [better_depth, max(ratio_list)]
    results_tree.append(dta_results_tree)
    
ent_gini_choice_list = np.array(list(criterion_best.values()))
better_ratio = list(ent_gini_choice_list[:,1])
better_index = better_ratio.index(max(better_ratio))
ent_or_gini  = criterion_chosen[better_index]
best_depth   = int(list(ent_gini_choice_list[:,0])[better_index])
print("The depth that gives the best right-to-total ratio is: ", ent_or_gini, ", ", best_depth)
    
    
#use test data to construct conf_matrix
dta_results_tree = results_tree[better_index]
conf_matrix_tree = np.zeros([5,5])
for i in range(5):
    for j in range(5):
        conf_matrix_tree[i,j] = np.sum((dta_results_tree[dta_results_tree.inx_test].N_stars == i+1) * (dta_results_tree[dta_results_tree.inx_test][best_depth-2] == j+1))


print("")
print("We use test data to construct a confusion_matrix: ")
print(conf_matrix_tree)
right_tree = conf_matrix_tree.diagonal(offset = 0).sum()
total_tree = conf_matrix_tree.sum()
ratio_tree = round(right_tree/total_tree,4)
print(ratio_tree)

right_to_total_dict["Trees"] = ratio_tree


#%%%% KNN
max_k_nn = 11
k_dict_knn = {}
results_list_knn = []

# Do for 1 to 10 neighbors
for k in range(1,max_k_nn):
    clf      = KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)
    results_list.append(
            np.concatenate(
                    [
                            clf.predict(X_train),
                            clf.predict(X_valid),
                            clf.predict(X_test )
                    ]))
    
   
    dta_results_knn              = pd.DataFrame(results_list).transpose()
    dta_results_knn.loc[dta_results_knn[0]>5,0] = 5
    dta_results_knn.loc[dta_results_knn[0]<1,0] = 1
    dta_results_knn['inx_train'] = inx_train.to_list()
    dta_results_knn['inx_valid'] = inx_valid.to_list()
    dta_results_knn['inx_test']  = inx_test.to_list()
    dta_results_knn["N_stars"] = dta.N_stars.copy()

    
    #Confusion matrix
    conf_matrix4 = np.zeros([5,5])
    for i in range(5):
        for j in range(5):
            conf_matrix4[i,j] = np.sum((dta_results_knn[dta_results_knn.inx_valid].N_stars == i+1)*(dta_results_knn[dta_results_knn.inx_valid][k-1] == j+1))
            
    right = conf_matrix4.diagonal(offset = 0).sum()
    total = conf_matrix4.sum()
    
    #Add k and its right-to-total ratio to the dictionary
    k_dict_knn[k] = round(right/total,4)


# print dictionary and find the smallest best k
print(k_dict_knn)

ratio_list_knn = list(k_dict_knn.values())
max_index_knn = ratio_list_knn.index(max(ratio_list_knn))
best_k = list(k_dict_knn.keys())[max_index_knn]
print("The k that gives the best right-to-total ratio is: ", best_k)

#use test data to construct conf_matrix

conf_matrix_knn = np.zeros([5,5])
for i in range(5):
    for j in range(5):
        conf_matrix_knn[i,j] = np.sum((dta_results_knn[dta_results_knn.inx_test].N_stars == i+1) * (dta_results_knn[dta_results_knn.inx_test][best_k-1] == j+1))

print("")
print(conf_matrix_knn)
right_knn = conf_matrix_knn.diagonal(offset = 0).sum()
total_knn = conf_matrix_knn.sum()
ratio_knn = round(right_knn/total_knn,4)
print(ratio_knn)

right_to_total_dict["KNN"] = ratio_knn



#%%%% Lasso
clf5 = linear_model.Lasso(alpha=0.1)
clf5.fit(X_train, Y_train)

dta['N_star_hat_lasso']             = np.concatenate(
        [
                clf5.predict(X_train.toarray()),
                clf5.predict(X_valid.toarray()),
                clf5.predict(X_test.toarray( ))
        ]).round().astype(int)

dta.loc[dta['N_star_hat_lasso']>5,'N_star_hat_lasso'] = 5
dta.loc[dta['N_star_hat_lasso']<1,'N_star_hat_lasso'] = 1

# confusion matrix for lasso
conf_matrix5 = np.zeros([5,5])
for i in range(5):
    for j in range(5):
        conf_matrix5[i,j] = np.sum((dta[inx_test].N_stars == i+1)*(dta[inx_test].N_star_hat_lasso == j+1))    

print(conf_matrix5)

# print the probability of right(almost right) prediction
right = conf_matrix5.diagonal(offset = 0).sum()
total = conf_matrix5.sum()
right_to_total = round(right/total,4)
print(right_to_total)

right_to_total_dict["Lasso"] = right_to_total


#%%%% Compare which classifier is the best here
print(right_to_total_dict)

# predictive model that gives the best oos prediction
best_mod = [model for model, value in right_to_total_dict.items() if value == max(right_to_total_dict.values())]
        
# print the maximum right_to_total
print("")
max_rrt = max(right_to_total_dict.values())

print("From the results above, ", end = "") 
print(*best_mod, sep = ", ", end=" ")
print("give(s) us the best prediction, and the right-to-total ratio is {}.".format(max_rrt))


