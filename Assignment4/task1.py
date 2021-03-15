import pandas as pd
import numpy as np
from math import log
import pydot
import graphviz

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

def same_class(examples, lim): #return Bool
    a = examples['Survived'].to_numpy()

    if (a[0]==a).mean() > lim or (a[0]==a).mean() < (1-lim):
        return True
    else:
        return False
    #return (a[0]==a).all() # works

def decisionTreeLearning(examples, attributes, default, simplify):
    #print(f"default: {default}")
    
    if len(examples) == 0:
        #print("empty examples")
        return default
    elif same_class(examples, 1):
        #print(f"same class {examples['Survived'].to_numpy()[0]}")
        return examples['Survived'].to_numpy()[0]
    elif len(attributes) == 0:
        #print("empty attributes")
        return mode(examples)

    else:
        best, split = choose_attribute(attributes, examples, 'infoGain')
        tree = {best:{}}
        attributes_new = list(attributes)
        attributes_new.remove(best) # Denne kan endre attributes globalt. Lag heller kopi
        if split == 0:
            values, _ = np.unique(examples[best],return_counts=True)
            for value in values:
                #print(f"testing for {value} of {best}\n Attributes left: {attributes_new}")
                examples_with_value = examples.where(examples[best]==value).dropna()
                subtree = decisionTreeLearning(examples_with_value,attributes_new,mode(examples), simplify)
                if canSimplify(subtree) and simplify:
                    value2 = subtree[list(subtree)[0]]
                    value2 = value2[list(value2)[0]]
                    subtree = value2
                tree[best][value] = subtree

        else:
            examples_under_split = examples.where(examples[best] < split).dropna()
            subtree = decisionTreeLearning(examples_under_split,attributes_new,mode(examples), simplify)
            if canSimplify(subtree) and simplify:
                value2 = subtree[list(subtree)[0]]
                value2 = value2[list(value2)[0]]
                subtree = value2
            tree[best][f"under {split}"] = subtree
            
            examples_over_split = examples.where(examples[best] > split).dropna()
            subtree = decisionTreeLearning(examples_over_split,attributes_new,mode(examples), simplify)
            if canSimplify(subtree) and simplify:
                value2 = subtree[list(subtree)[0]]
                value2 = value2[list(value2)[0]]
                subtree = value2
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


def canSimplify(tree):

    if tree == 0 or tree == 1:
        return False

    attribute = list(tree)[0]
    nextTree = tree[attribute]
    attributes = list(nextTree)

    value = nextTree[attributes[0]]
    #print(f"\ntree {tree} \n value: {value}")

    #print(nextTree)
    count = 0

    for attribute in attributes:
        #print(f"attribute: {attribute}\n NextTree[att]= {nextTree[attribute]}")
        if nextTree[attribute] == value:
            count += 1
    
    if count == len(attributes):
        #print(f"Can simplify. \n {attributes}")
        return True



def draw(parent_name, child_name):
    edge = pydot.Edge(parent_name, child_name)
    graph.add_edge(edge)

def visit(node, parent=None):
    for k,v in node.items():
        if isinstance(v, dict):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if parent:
                draw(str(parent), str(k))
            visit(v, k)
        else:
            draw(str(parent), str(k))
            # drawing the label using a distinct name
            draw(str(k), str(k)+'_'+str(v))

"""
def formulate(tree, parent):

    attribute = list(tree)[0]
    nextTree = tree[attribute]
    attributes = list(nextTree)
    value = nextTree[attributes[0]]
    

    for att in attributes:
        if parent == None:
            subtree2 = 0
        else:
            subtree2 = 0
        #subtree = formulate()
        subtree=1
        print()
        
        tree[attribute][att] = subtree

"""



if __name__ == "__main__":

    attributes_disc = ['Pclass','Sex','Embarked']
    usedCols_disc = ['Survived','Pclass','Sex','Embarked']

    attributes_cont = ['Pclass','Sex','SibSp','Parch','Fare','Embarked']
    usedCols_cont = ['Survived','Pclass','Sex','SibSp','Parch','Fare','Embarked']

    attributes_test = ['Fare']
    usedCols_test = ['Survived','Fare']

    simplifyBool = True
    includeContinous = False

    if includeContinous:
        train_df = pd.read_csv(trainFile, usecols=usedCols_cont)
        test_df = pd.read_csv(testFile, usecols=usedCols_cont)

        tree = decisionTreeLearning(train_df, attributes_cont, None, simplifyBool)
        print(tree)

        simpleTree = {'Sex': {'female': 1.0, 'male': 0.0}}

        accuracy = testTree(test_df, tree)
        print(f'Accuracy: {accuracy} %')

    else:
        train_df = pd.read_csv(trainFile, usecols=usedCols_disc)
        test_df = pd.read_csv(testFile, usecols=usedCols_disc)

        tree = decisionTreeLearning(train_df, attributes_disc, None, simplifyBool)
        print("tree: ",tree)

        #newTree = formulate(tree,None)
        tree_copy = tree.copy()
        graph = pydot.Dot(graph_type='graph')
        visit(tree_copy)
        print(graph.to_string())
        graph.write_png('graph.png')



        simpleTree = {'Sex': {'female': 1.0, 'male': 0.0}}

        #newTree = simplifyTree(tree)

        accuracy = testTree(test_df, tree)
        print(f'Accuracy: {accuracy} %')



    """
    tester = pd.read_csv(smallTestFile, usecols=usedCols_test)

    tree1 = decisionTreeLearning(tester, attributes_test, None)

    print(tree)

    accuracy = testTree(test_df, tree)
    print(f'Accuracy: {accuracy} %')
    """



    