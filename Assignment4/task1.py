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
        splittingPoints = np.zeros(len(attributes))
        entropies = np.zeros(len(attributes))
        entropyNow = entropy(examples['Survived']) #trenger man denne? Blir jo bare et tall uans
        #print("now: ",entropyNow)
        j=0
        for attribute in attributes:
            entropyAfter = 0

            values, counts = np.unique(examples[attribute],return_counts=True)
            #print(attribute, values, counts)

            if len(values) > 10: # continous variable
                numSurvived = 0
                numSurvivedTotal = np.sum(examples['Survived'])
                k = 0
                countsUnder = 0
                entropiesBetweenValues = np.ones(len(values))

                while numSurvived < numSurvivedTotal/2:
                    survivedCol = examples.where(examples[attribute]==values[k]).dropna()['Survived']

                    numSurvived += np.sum(survivedCol)
                    countsUnder += counts[j]


                    splitting_point_value = (values[k]+values[k+1]) / 2
                    survivedColUnderSplit = examples.where(examples[attribute] < splitting_point_value).dropna()['Survived']
                    survivedColOverSplit = examples.where(examples[attribute] > splitting_point_value).dropna()['Survived']
                    entropyUnder = countsUnder/np.sum(counts)*entropy(survivedColUnderSplit)
                    entropyOver = (np.sum(counts)-countsUnder)/np.sum(counts) * entropy(survivedColOverSplit)
                    entropyTotal = entropyUnder + entropyOver
                    entropiesBetweenValues[k] = entropyTotal

                    #print(f"k= {k} entropy= {entropyTotal}")

                    k+=1
                
                maxIndex = np.argmin(entropiesBetweenValues)
                splittingPoints[j] = (values[maxIndex]+values[maxIndex+1])/2
                entropyAfter = entropiesBetweenValues[maxIndex]
                #print(f"maxIndex= {maxIndex} entropy= {entropiesBetweenValues[maxIndex]} split= {splittingPoints[j]}")
            
            
            else:  

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
        return attributes[maxIndex], splittingPoints[maxIndex]

    return True

def mode(examples):
    values, counts = np.unique(examples['Survived'],return_counts=True)
    maxIndex = np.argmax(counts,axis=0)
    return values[maxIndex]

def same_class(examples): #return Bool
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
        best, split = choose_attribute(attributes, examples, 'infoGain')
        tree = {best:{}}
        attributes.remove(best)
        if split == 0:
            values, _ = np.unique(examples[best],return_counts=True)
            for value in values:
                examples_with_value = examples.where(examples[best]==value).dropna()
                subtree = decisionTreeLearning(examples_with_value,attributes,mode(examples))
                tree[best][value] = subtree

        else:
            examples_under_split = examples.where(examples[best] < split).dropna()
            subtree = decisionTreeLearning(examples_under_split,attributes,mode(examples))
            tree[best][f"under {split}"] = subtree
            
            examples_over_split = examples.where(examples[best] > split).dropna()
            subtree = decisionTreeLearning(examples_over_split,attributes,mode(examples))
            tree[best][f"over {split}"] = subtree

    return tree

def predict(row, tree):
    attribute = list(tree)[0]
    nextTree = tree[attribute]
    value = row[attribute]

    if attribute != 'Fare': #endre!!!!!!!! Gjør mer generelt.
        nextTree = nextTree[value]
    else:
        split = float(list(nextTree.keys())[0].split()[1])
        if value < split:
            nextTree = nextTree[list(nextTree.keys())[0]]
        else:
            nextTree = nextTree[list(nextTree.keys())[1]] # ikke sikkert disse funker for alle. kommer an på rekkefølgen av over/under i treet
        

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
    # Possibly bug in accuracy prediction. Is same if i remove Fare and Embarked.. May also be because Fare info i collected by Pclass
    return accuracy


if __name__ == "__main__":

    attributes_disc = ['Pclass','Sex','SibSp','Parch','Fare','Embarked']
    usedCols_disc = ['Survived','Pclass','Sex','SibSp','Parch','Fare','Embarked']

    attributes_test = ['Fare']
    usedCols_test = ['Survived','Fare']


    train_df = pd.read_csv(trainFile, usecols=usedCols_disc)
    test_df = pd.read_csv(testFile, usecols=usedCols_disc)

    tester = pd.read_csv(smallTestFile, usecols=usedCols_test)

    #tree1 = decisionTreeLearning(tester, attributes_test, None)

    tree = decisionTreeLearning(train_df, attributes_disc, None)
    print(tree)

    accuracy = testTree(test_df, tree)
    print(f'Accuracy: {accuracy} %')



    