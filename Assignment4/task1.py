import pandas as pd
import numpy as np
from math import log
import pydot
import graphviz

trainFile = 'Assignment4/train.csv'
testFile = 'Assignment4/test.csv'
smallTestFile = 'Assignment4/testing.csv'

# Returns entropy of column
def entropy(col):
    p = pd.Series(col).value_counts(normalize=True)
    return -(p * np.log(p)/np.log(2)).sum()

# Choosing the best attribute
def choose_attribute(attributes, examples, listContinousAtt):

    splittingPoints = np.zeros(len(attributes))
    entropies = np.zeros(len(attributes))
    entropyNow = entropy(examples['Survived'])
    j=0
    for attribute in attributes:
        entropyAfter = 0
        values, counts = np.unique(examples[attribute],return_counts=True)

        if attribute in listContinousAtt:
            #Continous Attribute
            numSurvived = 0
            k = 0
            countsUnder = 0
            entropiesBetweenValues = np.ones(len(values))

            while len(values) > k+1:
                survivedCol = examples.where(examples[attribute]==values[k]).dropna()['Survived']

                numSurvived += np.sum(survivedCol)
                countsUnder += counts[k]

                splitting_point_value = (values[k]+values[k+1]) / 2
                survivedColUnderSplit = examples.where(examples[attribute] < splitting_point_value).dropna()['Survived']
                survivedColOverSplit = examples.where(examples[attribute] > splitting_point_value).dropna()['Survived']
                entropyUnder = countsUnder/np.sum(counts)*entropy(survivedColUnderSplit)
                entropyOver = (np.sum(counts)-countsUnder)/np.sum(counts) * entropy(survivedColOverSplit)
                entropyTotal = entropyUnder + entropyOver
                entropiesBetweenValues[k] = entropyTotal

                k+=1
            
            maxIndex = np.argmin(entropiesBetweenValues)
            if len(values)==1:
                splittingPoints[j] = np.round(values[0],3)
            else:
                splittingPoints[j] = np.round((values[maxIndex]+values[maxIndex+1])/2,3)
            entropyAfter = entropiesBetweenValues[maxIndex]        
        
        else:  
            #Categorical Attribute
            for i in range(len(values)):
                col_to_cal_entropy = examples.where(examples[attribute]==values[i]).dropna()['Survived']
                entropyI = entropy(col_to_cal_entropy)
                entropyAfter += (counts[i]/np.sum(counts))*entropyI

        entropies[j] = entropyNow-entropyAfter
        j+=1
    
    maxIndex = np.argmax(entropies,axis=0)

    return attributes[maxIndex], splittingPoints[maxIndex]

# Returning the Survived class that is mostly represented
def mode(examples):
    values, counts = np.unique(examples['Survived'],return_counts=True)
    maxIndex = np.argmax(counts,axis=0)
    return values[maxIndex]

#If lim=1 -> Returns True only if all are same class. Otherwise returns true if at least lim*100 % are same class
def same_class(examples, lim): #return Bool
    a = examples['Survived'].to_numpy()

    if (a[0]==a).mean() >= lim or (a[0]==a).mean() <= (1-lim):
        return True
    else:
        return False

# Create a decision Tree
def decisionTreeLearning(all_examples, examples, attributes, default, simplify, listContinousAtt):
    
    if len(examples) == 0:
        return default
    elif same_class(examples, 1):
        return examples['Survived'].to_numpy()[0]
    elif len(attributes) == 0:
        return mode(examples)

    else:
        best, split = choose_attribute(attributes, examples, listContinousAtt)
        tree = {best:{}}
        attributes_new = list(attributes)
        attributes_new.remove(best) # If i dont copy, the attributes values will be changed globally
        if split==0 and best not in listContinousAtt:
            #Categorical Attribute
            values = np.unique(all_examples[best])
            
            for value in values:
                examples_with_value = examples.where(examples[best]==value).dropna()
                subtree = decisionTreeLearning(all_examples,examples_with_value,attributes_new,mode(examples), simplify, listContinousAtt)
                if canSimplify(subtree) and simplify:
                    value2 = subtree[list(subtree)[0]]
                    value2 = value2[list(value2)[0]]
                    subtree = value2
                tree[best][value] = subtree

        else:
            #Continous Attribute
            examples_under_split = examples.where(examples[best] <= split).dropna()
            subtree = decisionTreeLearning(all_examples,examples_under_split,attributes_new,mode(examples), simplify, listContinousAtt)
            if canSimplify(subtree) and simplify:
                value2 = subtree[list(subtree)[0]]
                value2 = value2[list(value2)[0]]
                subtree = value2
            tree[best][f"under {split}"] = subtree
            
            examples_over_split = examples.where(examples[best] > split).dropna()
            subtree = decisionTreeLearning(all_examples,examples_over_split,attributes_new,mode(examples), simplify, listContinousAtt)
            if canSimplify(subtree) and simplify:
                value2 = subtree[list(subtree)[0]]
                value2 = value2[list(value2)[0]]
                subtree = value2
            tree[best][f"over {split}"] = subtree

    return tree


# Predicting if a person survived
def predict(row, tree, listContinousAtt):

    attribute = list(tree)[0]
    nextTree = tree[attribute]
    value = row[attribute]

    if attribute not in listContinousAtt or len(nextTree) == 1:
        nextTree = nextTree[value]
    else:
        split = float(list(nextTree.keys())[0].split()[1])
        if value <= split:
            nextTree = nextTree[list(nextTree.keys())[0]]
        else:
            nextTree = nextTree[list(nextTree.keys())[1]]
        
    if nextTree == 0 or nextTree == 1:
        return nextTree
    else:
        return predict(row, nextTree, listContinousAtt)

# Testing tree and returning accuracy
def testTree(examples, tree, listContinousAtt):
    rows = examples.iloc[:,1:].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["pred"]) 
    
    for i in range(len(examples)):
        predicted.loc[i,"pred"] = predict(rows[i], tree, listContinousAtt)

    accuracy = np.sum(predicted["pred"] == examples["Survived"])/len(examples)*100

    return accuracy

# Function used in descionTreeLearning() to check if tree can be simplifyed. 
# I.e if all the leaf nodes have same value
def canSimplify(tree):

    if tree == 0 or tree == 1:
        return False

    attribute = list(tree)[0]
    nextTree = tree[attribute]
    attributes = list(nextTree)
    value = nextTree[attributes[0]]

    count = 0
    for attribute in attributes:
        if nextTree[attribute] == value:
            count += 1
    
    if count == len(attributes):
        #Can simplify!
        return True

#Function used in draw
def findParentId(name):
    edges = list(graph.obj_dict['edges'])
    i_corr = 0
    j_corr = 0
    found = False
    for i in range(len(edges)):
        for j in range(len(edges[i])):
            if edges[i][j] == id(name)+edges[i][j-1]:
                #Found a node with name equal to parent node!
                found = True
                i_corr = i
                j_corr = j
    
    if found:            
        return edges[i_corr][j_corr-1]+id(name)
    else:
        return id(name)

#Inspired by Stackoverflow user 'greeness' (https://stackoverflow.com/questions/13688410/dictionary-object-to-decision-tree-in-pydot)
def draw(parent_name, child_name):
    parentId = findParentId(parent_name)
    graph.add_node(pydot.Node(parentId, label=str(parent_name)))
    graph.add_node(pydot.Node(id(child_name)+parentId, label=str(child_name)))
    
    edge = pydot.Edge(parentId, id(child_name)+parentId)
    graph.add_edge(edge)


#Inspired by Stackoverflow user 'greeness' (https://stackoverflow.com/questions/13688410/dictionary-object-to-decision-tree-in-pydot)
def visit(node, parent=None):
    for k,v in node.items():
        if isinstance(v, dict):
            if parent:
                draw(parent, k)
            visit(v, k)
        else:
            draw(parent, k)
            draw(k, v)



if __name__ == "__main__":

    attributes_disc = ['Pclass','Sex','Embarked']
    usedCols_disc = ['Survived','Pclass','Sex','Embarked']

    attributes_cont = ['Pclass','Sex','SibSp','Parch','Fare','Embarked']
    usedCols_cont = ['Survived','Pclass','Sex','SibSp','Parch','Fare','Embarked']

    listContinousAtt = ['SibSp','Parch','Fare']


    #### CHANGE THE VARIABLES BELOW TO TEST DIFFERENT CASES ####
    simplifyBool = False
    includeContinous = True



    if includeContinous:
        filename = 'graph_continous_full.png'
        if simplifyBool:
            filename = 'graph_continous_simple.png'

        train_df = pd.read_csv(trainFile, usecols=usedCols_cont)
        test_df = pd.read_csv(testFile, usecols=usedCols_cont)



        print(f"\nStarting training on continous data set.")
        tree = decisionTreeLearning(train_df, train_df, attributes_cont, None, simplifyBool, listContinousAtt)

        tree_copy = tree.copy()
        graph = pydot.Dot(graph_type='graph')
        visit(tree_copy)
        graph.write_png(filename)

        print(f"\nFinished building the tree. Saved the image to '{filename}'. Simplifyed? {simplifyBool}")
        print('\nTesting the tree on our test data')

        accuracy = testTree(test_df, tree, listContinousAtt)
        print(f'Finished testing. Accuracy: {accuracy} %')

    else:
        filename = 'graph_categorical_full.png'
        if simplifyBool:
            filename = 'graph_categorical_simple.png'

        train_df = pd.read_csv(trainFile, usecols=usedCols_disc)
        test_df = pd.read_csv(testFile, usecols=usedCols_disc)

        print(f"\nStarting training on categorical data set.")

        tree = decisionTreeLearning(train_df, train_df, attributes_disc, None, simplifyBool, listContinousAtt)

        tree_copy = tree.copy()
        graph = pydot.Dot(graph_type='graph')
        visit(tree_copy)
        graph.write_png(filename)


        print(f"\nFinished building the tree. Saved the image to '{filename}'. Simplifyed? {simplifyBool}")
        print('\nTesting the tree on our test data')

        accuracy = testTree(test_df, tree, listContinousAtt)
        print(f'Finished testing. Accuracy: {accuracy} %')



    