# Code framework courtesy of Jason Brownlee
# https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

import csv
import random
import math
import sys
sys.path.append("NumPy_path")
import numpy as np
import pandas as pd
from random import randrange
from statistics import mode
import pickle
import _thread # For use with Oscar - Brown University's Supercomputer
import concurrent.futures

from platform import python_version
print(python_version())

#Load data from csv file
def loadCsv(filename):
    lines = csv.reader(open(filename, newline=''), delimiter=',', quotechar='|')
    next(lines)
    dataset = []
    for row in lines:
        dataset.append([float(x) for x in row])
    return dataset

#Split data into training and validation portions
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

#Split data into training, testing, and validation portions
def splitDatasetThreeWays(dataset, trainingRatio, testingRatio):
    trainSize = int(len(dataset) * trainingRatio)
    trainSet = []
    testSize = int(len(dataset) * testingRatio)
    testSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    while len(testSet) < testSize:
        index = random.randrange(len(copy))
        testSet.append(copy.pop(index))
    return [trainSet, testSet, copy]

#Split data into K Folds
def splitDatasetKFolds(dataset, numberOfFolds):
    datasetSplit = list()
    dataset_copy = list(dataset)
    foldSize = math.floor(len(dataset) / numberOfFolds)
    for i in range(numberOfFolds):
        fold = list()
        while len(fold) < foldSize:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        datasetSplit.append(fold)
    return datasetSplit

#Split data into K Folds (using DataFrames)
def splitDatasetKFolds_df(dataset, numberOfFolds):
    datasetSplit = list()
    foldSize = int(len(dataset.index)) / numberOfFolds
    indices_left = list(range(len(dataset.index)))
    for i in range(numberOfFolds):
        fold = list()
        while len(fold) < foldSize:
            index = random.choice(indices_left)
            fold.append(indices_left.pop(index))
        dataSplit.append(dataset[fold])
    return datasetSplit

#Group data points by class
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

#Compute the mean across a collection of numbers
def mean(numbers):
    return sum(numbers)/float(len(numbers))

#Compute the standard deviation across a collection of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

#Compute summary statistics for the entire dataset
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

#Compute summary statistics per class
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

#Compute per-attribute probabilities
def calculateProbability(x, mean, stdev):
    if stdev == 0:
        return 0.000001
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return max(0.000001,(1 / (math.sqrt(2*math.pi) * stdev)) * exponent)

#Calculate global calculateProbability
#To prevent data type underflow, we use log probabilities
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 0
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] += math.log10(calculateProbability(x, mean, stdev))
    return probabilities

#Determine the most likely class label
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

#Make predictions on the unseen data
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

#Determine accuracy percentage of a prediction
def accuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i] == predictions[i]:
            correct += 1
    return correct / float(len(testSet)) * 100.0

#Compute performance in terms of F1 scores
def evaluate(testSet, predictions):
    tp1 = 0.0
    fp1 = 0.0
    fn1 = 0.0
    tp0 = 0.0
    fp0 = 0.0
    fn0 = 0.0
    for i in range(len(testSet)):
        if testSet[i][-1] == 0:
            if predictions[i] == 0:
                tp0 += 1.0
            else:
                fp1 += 1.0
                fn0 += 1.0
        else:
            if predictions[i] == 1:
                tp1 += 1.0
            else:
                fp0 += 1.0
                fn1 += 1.0
    p0 = tp0/(tp0+fp0)
    p1 = tp1/(tp1+fp1)
    r0 = tp0/(tp0+fn0)
    r1 = tp1/(tp1+fn1)
    print("F1 (Class 0): "+str((2.0*p0*r0)/(p0+r0)))
    print("F1 (Class 1): "+str((2.0*p1*r1)/(p1+r1)))

#Compute performance in terms of F1 scores
#Returns the F1 scores for both classes
def evaluate_return(testSet, predictions):
    tp1 = 0.0
    fp1 = 0.0
    fn1 = 0.0
    tp0 = 0.0
    fp0 = 0.0
    fn0 = 0.0
    for i in range(len(testSet)):
        if testSet[i] == 0:
            if predictions[i] == 0:
                tp0 += 1.0
            else:
                fp1 += 1.0
                fn0 += 1.0
        else:
            if predictions[i] == 1:
                tp1 += 1.0
            else:
                fp0 += 1.0
                fn1 += 1.0
    p0 = tp0/(tp0+fp0)
    p1 = tp1/(tp1+fp1)
    r0 = tp0/(tp0+fn0)
    r1 = tp1/(tp1+fn1)
    return((2.0*p0*r0)/(p0+r0), (2.0*p1*r1)/(p1+r1)) #F1 for class 0, F1 for class 1

#Run the end-to-end baseline system
def baseline():
    random.seed(13)
    filename = 'readmission.csv'
    splitRatio = 0.80
    dataset = loadCsv(filename)
    trainingSet, valSet = splitDataset(dataset, splitRatio)
    print('Split '+str(len(dataset))+' rows into train = '+str(len(trainingSet))+' and test = '+str(len(valSet))+' rows.\n')
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, valSet)
    evaluate(valSet, predictions)


#Master function for testing random forests
def k_fold_val(num_trees, num_nodes,k_value,filename,subset=500):

    # Below are varying sized datasets used to test the functionality of training()
    # This was in an effort to ensure our model was working as expected without significant run-time complexity

    #dataSet = loadCsv('simpleset.csv')
    #dataSet = loadCsv('test_mix.csv')
    #dataSet = loadCsv('readmission.csv')

    dataSet = loadCsv(filename)

    # Split data set into KFolds
    fold_set = splitDatasetKFolds(dataSet, k_value)

    # Initialize class F1 score lists
    class0_F1 = []
    class1_F1 = []

    for i in range(len(fold_set)):
        test_i = i
        train_i = list(range(len(fold_set)))
        train_i.pop(i)
        
        train_data = []
        for index in train_i:
            train_data.extend(fold_set[index])

        #Over sampling function took extremely long to run - deferring to under sampling.
        #train_oversampled = overSample(train_data)
        train_undersampled = underSample(train_data)
        
        train_df = pd.DataFrame(train_undersampled)
        train_x = train_df.iloc[:,:-1]
        train_y = train_df.iloc[:,-1]
        
        test_df = pd.DataFrame(fold_set[test_i])
        test_x = test_df.iloc[:,:-1]
        test_y = test_df.iloc[:,-1]

        #Utilize our RandomForest classifier with predetermined number of trees and nodes.
        rf = RandomForest(num_trees,num_nodes) #numtrees, maxnodes
        #rf.fit_multi(x_folds[train_i],y_folds[train_i])
        #rf.AdaBoost_calc_weights(rf.predictions(x_folds[train_i]),y_folds[train_i])
        #rf.fit_multi(train_x,train_y)
        rf.fit_multi_subset(fold_set[test_i],subset) #this takes in dataset (list of lists) instead of df now
        rf.AdaBoost_calc_weights(rf.predictions(train_x),train_y) #calculating predictions on the whole train fold

        data = rf
        f = open("rf"+str(i), 'wb')
        pickle.dump(data, f)
        f.close()

        #Actual Training
        #predictions = rf.AdaBoost_predictions(x_folds[test_i])
        #(class0,class1) = evaluate_return(y_folds[test_i], predictions)
        predictions = rf.AdaBoost_predictions(test_x)
        (class0,class1) = evaluate_return(test_y, predictions)

        #Check f1 value for trained and saved models
#        predictions = rf.AdaBoost_predictions(test_set_x)
#        (class0,class1) = evaluate_return(test_set_y, predictions)

        class0_F1.append(class0)
        class1_F1.append(class1)
        f = open("f1", 'a')
        temp_str = "With"+str(num_trees)+" trees and "+str(num_nodes)+" nodes: class 0 F1: "+str(class0)+" class 1 F1: "+str(class1)
        f.write(temp_str)
        f.write("\n")
        f.close()

    print('With ',num_trees,' trees and ',num_nodes,' nodes:','class 0 F1: ',np.mean(class0_F1),' class 1 F1: ',np.mean(class1_F1))

###
def training(training_set):

        train_df = pd.DataFrame(training_set)
        train_x = train_df.iloc[:,:-1]
        train_y = train_df.iloc[:,-1]
        
        # Warning we expect this will take a long time
        # We do print statements everytime a tree is finished
        num_trees = 200
        num_nodes = 30
        subset = 300
        
        #Utilize our RandomForest classifier with predetermined number of trees and nodes.
        rf = RandomForest(num_trees,num_nodes) #numtrees, maxnodes
        rf.fit_subset(training_set,subset) #this takes in dataset (list of lists) instead of df now
        
        # Trained model is stored here
        data = rf
        f = open("rf"+str(i), 'wb')
        pickle.dump(data, f)
        f.close()

#  Attempts to speed up runtime using multiple threads
def multiTrain():
    _thread.start_new_thread(training, ("rf1", ))
    _thread.start_new_thread(training, ("rf2", ))

#Load a trained model from file
def loading():
    #return a whole forest
    f = open('rf', 'rb')
    data = pickle.load(f)
    f.close()
    return data


#Use the trained model to produce predictions for our dataset
def inference(model, data):
    #depends if label column is given or not:
    test_set = pd.DataFrame(data)
    return model.mode_predictions(test_set)
    pass


# ## Random Forest Implementation ####

def gini_helper(x,y):
    classes = {}
    for i in range(len(y)):
        if y.iloc[i] in classes.keys():
            classes[y.iloc[i]] += 1
        else:
            classes[y.iloc[i]] = 1
    counter = 0
    for c in classes.keys():
        counter += (classes[c] / len(y))**2
    return 1 - counter

def Gini(left_x,right_x,y):
    left_gini = gini_helper(left_x,y)
    right_gini = gini_helper(right_x,y)
    return left_gini*len(left_x)/len(y) + right_gini*len(right_x)/len(y)

# Node class utilizing Gini impurity to find and make optimal split on left and right child nodes
class Node(): #have as properties a left node and a right node - can either be another node (move down tree) or an output class (if at end of branch)
    def __init__(self,x,y,tree,print_num=False):
        self.x = x #data
        self.y = y #data
        self.tree = tree #tree it's a part of
        self.tree.num_nodes += 1 #let tree know you've made a new node
        if print_num:
            print("current node number:")
            print(self.tree.num_nodes)

        self.left_child = None #next node or output/prediction
        self.right_child = None #next node or output/prediction
        self.feature = None #feature it will make division by - alg finds best one
        self.threshold = None #where to divide by feature - alg will find best one
        self.gini = None

        self.left_x = []
        self.left_y = []
        self.right_x = []
        self.right_y = []

    def make_split(self,feature,threshold):
        left_indices = []
        right_indices = []
        for i in range(len(self.y)):
            if self.x[feature].iloc[i] <= threshold:
                left_indices.append(i)
            else:
                right_indices.append(i)
        left_x = self.x.iloc[left_indices]
        left_y = self.y.iloc[left_indices]
        right_x = self.x.iloc[right_indices]
        right_y = self.y.iloc[right_indices]
        return (left_x,left_y,right_x,right_y)

    def find_split(self):

        self.gini = 0.5 #we think this is the worst you can do

        #go through all possible features and threshold, calculate gini and compare to what we have stored - if better, store that feature, threshold, gini instead
        for feature in self.x.columns:
            x_sorted = self.x[feature].sort_values()
            for thresh in x_sorted:
                (left_x,left_y,right_x,right_y) = self.make_split(feature,thresh)
                new_gini = Gini(left_x,right_x,self.y)
                if new_gini <= self.gini and len(left_x) > 0 and len(right_x) > 0: #update values
                    self.feature = feature
                    self.threshold = thresh
                    self.gini = new_gini
                #if not, go on to next one

        #now have threshold and feature - set up children
        (self.left_x,self.left_y,self.right_x,self.right_y) = self.make_split(self.feature,self.threshold) #making the real split
    
        if self.gini == 0: #perfect division of classes - return labels/outputs
            self.left_child = self.left_y.iloc[0]
            self.right_child = self.right_y.iloc[0] #should all be the same because gini=0
        elif self.tree.num_nodes >= self.tree.maxnodes:
            self.left_child = mode(list(self.left_y))
            self.right_child = mode(list(self.right_y))
        else:
            if len(self.left_y) > 1:
                self.left_child = Node(self.left_x,self.left_y,self.tree)
                self.left_child.find_split()
            else:
                self.left_child = self.left_y.iloc[0]

            if len(self.right_y) > 1:
                self.right_child = Node(self.right_x,self.right_y,self.tree)
                self.right_child.find_split()
            else:
                self.right_child = self.right_y.iloc[0]

#  Class to make a single decision tree with the help of the predefined node class.
#  Builds out nodes starting at the root and returns completely built decision tree
class DecisionTree():
    def __init__(self,maxnodes=100):
        self.x = None
        self.y = None
        self.maxnodes = maxnodes
        self.num_nodes = 0
        self.root = None

    def fit(self,x,y):
        self.root = Node(x,y,self) #node_num might change
        self.root.find_split()

        return self

    def predict_point(self,x):
        node = self.root
        keepWalking = True
        while keepWalking:
            if x[node.feature] <= node.threshold:
                node = node.left_child
            else:
                node = node.right_child
            if not isinstance(node,Node):
                keepWalking = False
        return node #prediction

    def predictions(self,x_df):
        return [self.predict_point(row) for (i,row) in x_df.iterrows()]

# Random Forest Classifier which utilizes the DecisionTree class to build multiple trees of varying node sizes
class RandomForest():
    def __init__(self,num_trees,max_nodes):
        self.num_trees = num_trees
        self.max_nodes = max_nodes

        self.trees = [DecisionTree(max_nodes) for i in range(num_trees)]

    def fit(self,x,y):
        self.tree_features = [random.sample(x.columns.tolist(),int(np.floor(len(x.columns)/3))) for tree in self.trees]

        for i in range(self.num_trees):
            self.trees[i].fit(x.loc[:,self.tree_features[i]],y)

    # def fit_multi(self,x,y):
    #     self.tree_features = [random.sample(x.columns.tolist(),int(np.floor(len(x.columns)/3))) for tree in self.trees]
    #
    #     for i in range(self.num_trees):
    #         with concurrent.futures.ThreadPoolExecutor(max_workers = 2) as executor:
    #             thread = executor.submit(self.trees[i].fit, x.loc[:,self.tree_features[i]],y)
    #             #self.trees[i] = thread.result()
    #             print('tree number ',i+1,' trained')

    def fit_multi(self, x, y):
        self.tree_features = [random.sample(x.columns.tolist(),int(np.floor(len(x.columns)/3))) for tree in self.trees]
        with concurrent.futures.ThreadPoolExecutor(max_workers = 2) as executor:
            thread_list = []
            for i in range(self.num_trees):
                thread = executor.submit(self.trees[i].fit, x.loc[:,self.tree_features[i]],y)
                thread_list.append(thread)
            for future in concurrent.futures.as_completed(thread_list):
                print('tree number ',i+1,' trained')
    
    def fit_subset(self,dataset,subset=500):
        df = pd.DataFrame(dataset) #for selecting each tree's features
        df = df.iloc[:,:-1]
        self.tree_features = [random.sample(df.columns.tolist(),int(np.floor(len(df.columns)/3))) for tree in self.trees]
       

       
        for i in range(self.num_trees):
           
            # Split dataset by class to select a random subset of each
            separated_data = separateByClass(dataset)
            new_list = []
            new_list.extend(random.sample(separated_data[0],k=int(np.floor(subset/2)))) #half from majority class
            new_list.extend(random.sample(separated_data[1],k=int(np.floor(subset/2)))) #half from minory class
            random.shuffle(new_list)
       
            # Make dataframes
            df = pd.DataFrame(new_list)
            x = df.iloc[:,:-1]
            y = df.iloc[:,-1]
           
            self.trees[i].fit(x.loc[:,self.tree_features[i]],y)
            print('tree number ',i+1,' trained')
            
    def fit_multi_subset(self,dataset,subset=500):
        df = pd.DataFrame(dataset) #for selecting each tree's features
        df = df.iloc[:,:-1]
        self.tree_features = [random.sample(df.columns.tolist(),int(np.floor(len(df.columns)/3))) for tree in self.trees]
        
        # Split dataset by class to select a random subset of each
        separated_data = separateByClass(dataset)
        new_list = []
        new_list.extend(random.sample(separated_data[0],k=int(np.floor(subset/2)))) #half from majority class
        new_list.extend(random.sample(separated_data[1],k=int(np.floor(subset/2)))) #half from minory class
        random.shuffle(new_list)
        
        # Make dataframes
        df = pd.DataFrame(new_list)
        x = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers = 2) as executor:
            thread_list = []
            for i in range(self.num_trees):
                thread = executor.submit(self.trees[i].fit, x.loc[:,self.tree_features[i]],y)
                thread_list.append(thread)
            for future in concurrent.futures.as_completed(thread_list):
                print('tree number ',i+1,' trained')


    def predictions(self,x_df):
        final_predictions = []
        for (i,row) in x_df.iterrows():
            row_preds = [tree.predict_point(row) for tree in self.trees]
            final_predictions.append(row_preds)
        return final_predictions

    # Function which takes in a data frame of predictions and returns the mode from the
    def mode_predictions(self,x_df):
        pred_list = self.predictions(x_df)
        return [mode(l) for l in pred_list]

    # Function to calculate the weights of each feature using AdaBoost
    def AdaBoost_calc_weights(self,preds,true_labels):
        #calculate weights of classifiers
        preds_df = pd.DataFrame(preds)
        #first, calculate error of models with all data points weighted equally
        errors = []
        for col in preds_df.columns: #for each classifier
            column = preds_df.loc[:,col]
            errors.append(sum([1 for i in range(len(column)) if column[i] != true_labels[i]]) / len(true_labels) ) #preds is a list of lists
        #now calculate weights of models
        self.model_weights = [0.5*np.log((1-error)/error) if error != 0 else 1 for error in errors] #if error = 0 , set weight to one

        #can update weights of data points here

    # Function to make predictions on data with AdaBoost input
    def AdaBoost_predictions(self,x_df):
        preds_df = pd.DataFrame(self.predictions(x_df))
        print(self.model_weights)
        probs = sum([preds_df.iloc[:,i] * self.model_weights[i] for i in range(self.num_trees)]) / self.num_trees
        print(probs)
        return [1 if prob > 0.5 else 0 for prob in probs]

# Function to duplicate samples of readmitted patients (add data for minority class)
def overSample(data):
    separated_data = separateByClass(data)
    difference = (len(separated_data[0])-(len(separated_data[1])))
    new_list = []
    new_list.extend(separated_data[0])
    new_list.extend(separated_data[1])
    new_list.extend(random.choices(separated_data[1], k=difference))
    random.shuffle(new_list)
    best_list = separateByClass(new_list)
    new_difference = (len(best_list[0])-(len(best_list[1])))
    return new_list

# Function to delete a subset of samples of patients NOT readmitted (reduce data from majority class)
def underSample(data):
    separated_data = separateByClass(data)
    new_list = []
    new_list.extend(separated_data[1]) #all of minority class
    new_list.extend(random.sample(separated_data[0],k=len(separated_data[1]))) #sample without replacement from majority class
    random.shuffle(new_list)
    return new_list

# First pass of random forests classifier with AdaBoost and KFolds
# Obsolete function as we now under sample data
# Please refer to the training function for final implementation
def kfolds_run_and_calc(fold_set,num_trees,num_nodes):
    df_folds = [pd.DataFrame(fold) for fold in fold_set]
    #print(df_folds)
    x_folds = [df.iloc[:,:-1] for df in df_folds]
    y_folds = [df.iloc[:,-1] for df in df_folds]
    temp = loadCsv('readmission.csv')
    test_set = pd.DataFrame(temp)
    test_set_x = test_set.iloc[:,:-1]
    test_set_y = test_set.iloc[:,-1]
    class0_F1 = []
    class1_F1 = []
    for i in range(len(fold_set)):
        test_i = i
        train_i = list(range(len(fold_set))).pop(i)
        rf = RandomForest(num_trees,num_nodes) #numtrees,maxnodes
        rf.fit_multi(x_folds[train_i],y_folds[train_i])
        # rf.fit(x_folds[train_i], y_folds[train_i])
        rf.AdaBoost_calc_weights(rf.predictions(x_folds[train_i]),y_folds[train_i])
        predictions = rf.AdaBoost_predictions(test_set_x)
        (class0,class1) = evaluate_return(test_set_y, predictions)
        class0_F1.append(class0)
        class1_F1.append(class1)
        print('With ',num_trees,' trees and ',num_nodes,' nodes:','class 0 F1: ',class0,' class 1 F1: ',class1)
    print('With ',num_trees,' trees and ',num_nodes,' nodes:','class 0 F1: ',np.mean(class0_F1),' class 1 F1: ',np.mean(class1_F1))


training_set = loadCsv('readmission.csv')
model = training(training_set)
test_set = loadCsv('simpleset.csv')
predictions =inference(model, test_set)
evaluate(test_set, predictions)
