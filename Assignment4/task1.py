import pandas as pd
import numpy as np
from math import log

trainFile = 'Assignment4/train.csv'
testFile = 'Assignment4/test.csv'
smallTestFile = 'Assignment4/testing.csv'


def entropy(col):
    vc = pd.Series(col).value_counts(normalize=True, sort=False)
    return -(vc * np.log(vc)/np.log(2)).sum() #base 2 her, sant?


def choose_attribute(attributes, examples, type):

    if type == 'infoGain':
        #print("doing infoGain algo")
        entropies = np.zeros(len(attributes))
        entropyNow = entropy(examples['Survived']) #trenger man denne? Blir jo bare et tall uans
        #print("now: ",entropyNow)
        j=0
        for attribute in attributes:
            entropyAfter = 0

            values, counts = np.unique(examples[attribute],return_counts=True)
            #print(attribute, values, counts)

            for i in range(len(values)):
                col_to_cal_entropy = examples.where(examples[attribute]==values[i]).dropna()['Survived']
                #print(col_to_cal_entropy)
                entropyI = entropy(col_to_cal_entropy)
                entropyAfter += (counts[i]/np.sum(counts))*entropyI

            entropies[j] = entropyNow-entropyAfter
            j+=1
        
        maxIndex = np.argmax(entropies,axis=0)

        #print(maxIndex)
        #print(attributes[maxIndex])
        #print(entropies)
        return attributes[maxIndex]

    return True

def mode(examples):
    values, counts = np.unique(examples['Survived'],return_counts=True)
    maxIndex = np.argmax(counts,axis=0)
    return values[maxIndex]

def same_class(examples):
    a = examples['Survived'].to_numpy()
    return (a[0]==a).all() # works

def decisionTreeLearning(examples, attributes, default):
    
    if len(examples) == 0:
        #print("empty examples")
        return default
    elif same_class(examples):
        #print("same class")
        return examples['Survived'].to_numpy()[0]
    elif len(attributes) == 0:
        #print("empty attributes")
        return mode(examples)

    else:
        best = choose_attribute(attributes, examples, 'infoGain')
        tree = {best:{}}
        attributes.remove(best)
        values, _ = np.unique(examples[best],return_counts=True)

        for value in values:
            examples_with_value = examples.where(examples[best]==value).dropna()
            subtree = decisionTreeLearning(examples_with_value,attributes,mode(examples))
            tree[best][value] = subtree

    return tree

def predict(row, tree):
    attribute = list(tree)[0]
    nextTree = tree[attribute]

    value = row[attribute]
    nextTree = nextTree[value]

    if nextTree == 0 or nextTree == 1:
        return nextTree
    else:
        return predict(row, nextTree)



def testTree(examples, tree):
    #Removes the first col of examples, where Survived are
    rows = examples.iloc[:,1:].to_dict(orient = "records")
    
    predicted = pd.DataFrame(columns=["pred"]) 
    
    for i in range(len(examples)):
        predicted.loc[i,"pred"] = predict(rows[i],tree)

    accuracy = np.sum(predicted["pred"] == examples["Survived"])/len(examples)*100

    return accuracy


if __name__ == "__main__":

    attributes_disc = ['Pclass','Sex','SibSp','Parch']
    usedCols_disc = ['Survived','Pclass','Sex','SibSp','Parch']
    
    train_df = pd.read_csv(trainFile, usecols=usedCols_disc)
    test_df = pd.read_csv(testFile, usecols=usedCols_disc)

    tester = pd.read_csv(smallTestFile, usecols=usedCols_disc)


    tree = decisionTreeLearning(train_df, attributes_disc, None)
    print(tree)

    accuracy = testTree(test_df, tree)
    print(f'Accuracy: {accuracy} %')



    