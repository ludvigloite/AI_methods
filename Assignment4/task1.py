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
    return -(vc * np.log(vc)/np.log(2)).sum()


def choose_attribute(attributes, examples, type):

    listContinousAtt = ['SibSp','Parch','Fare']

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

            if attribute in listContinousAtt: #and len(values) > 1: # continous variable
                numSurvived = 0
                numSurvivedTotal = np.sum(examples['Survived'])
                k = 0
                countsUnder = 0
                entropiesBetweenValues = np.ones(len(values))

                while len(values) > k+1:
                    survivedCol = examples.where(examples[attribute]==values[k]).dropna()['Survived']

                    numSurvived += np.sum(survivedCol)
                    countsUnder += counts[k]


                    splitting_point_value = (values[k]+values[k+1]) / 2
                    #print(splitting_point_value)
                    survivedColUnderSplit = examples.where(examples[attribute] < splitting_point_value).dropna()['Survived']
                    survivedColOverSplit = examples.where(examples[attribute] > splitting_point_value).dropna()['Survived']
                    entropyUnder = countsUnder/np.sum(counts)*entropy(survivedColUnderSplit)
                    entropyOver = (np.sum(counts)-countsUnder)/np.sum(counts) * entropy(survivedColOverSplit)
                    entropyTotal = entropyUnder + entropyOver
                    entropiesBetweenValues[k] = entropyTotal

                    #print(f"k= {k} entropy= {entropyTotal}")

                    k+=1
                
                maxIndex = np.argmin(entropiesBetweenValues)
                if len(values)==1:
                    splittingPoints[j] = np.round(values[0],3)
                else:
                    splittingPoints[j] = np.round((values[maxIndex]+values[maxIndex+1])/2,3)
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

def decisionTreeLearning(all_examples, examples, attributes, default, simplify):
    #print(f"default: {default}")

    listContinousAtt = ['SibSp','Parch','Fare']
    
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
        if split==0 and best not in listContinousAtt:
            values, _ = np.unique(all_examples[best],return_counts=True) #kan kanskje ikke returnere counts?
            if len(values)==1:
                print(best)
            for value in values:
                #print(f"testing for {value} of {best}\n Attributes left: {attributes_new}")
                examples_with_value = examples.where(examples[best]==value).dropna()
                subtree = decisionTreeLearning(all_examples,examples_with_value,attributes_new,mode(examples), simplify)
                if canSimplify(subtree) and simplify:
                    value2 = subtree[list(subtree)[0]]
                    value2 = value2[list(value2)[0]]
                    subtree = value2
                tree[best][value] = subtree

        else:
            examples_under_split = examples.where(examples[best] < split).dropna()
            subtree = decisionTreeLearning(all_examples,examples_under_split,attributes_new,mode(examples), simplify)
            if canSimplify(subtree) and simplify:
                value2 = subtree[list(subtree)[0]]
                value2 = value2[list(value2)[0]]
                subtree = value2
            tree[best][f"under {split}"] = subtree
            
            examples_over_split = examples.where(examples[best] > split).dropna()
            subtree = decisionTreeLearning(all_examples,examples_over_split,attributes_new,mode(examples), simplify)
            if canSimplify(subtree) and simplify:
                value2 = subtree[list(subtree)[0]]
                value2 = value2[list(value2)[0]]
                subtree = value2
            tree[best][f"over {split}"] = subtree

    return tree

def predict(row, tree):

    listContinousAtt = ['SibSp','Parch','Fare']

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


def findParentId(name):
    edges = list(graph.obj_dict['edges'])
    i_corr = 0
    j_corr = 0
    found = False
    for i in range(len(edges)):
        for j in range(len(edges[i])):
            if edges[i][j] == id(name)+edges[i][j-1]:
                #print("found parent")
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
    #print(f"adding edge: {parentId} -- {id(child_name)+parentId}. id_child: {id(child_name)} {parent_name} -- {child_name}")
    
    edge = pydot.Edge(parentId, id(child_name)+parentId)
    graph.add_edge(edge)


#Inspired by Stackoverflow user 'greeness' (https://stackoverflow.com/questions/13688410/dictionary-object-to-decision-tree-in-pydot)
def visit(node, parent=None):
    for k,v in node.items():
        if isinstance(v, dict):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if parent:
                draw(parent, k)
            visit(v, k)
        else:
            draw(parent, k)
            # drawing the label using a distinct name
            draw(k, v)



if __name__ == "__main__":

    attributes_disc = ['Pclass','Sex','Embarked']
    usedCols_disc = ['Survived','Pclass','Sex','Embarked']

    attributes_cont = ['Pclass','Sex','SibSp','Parch','Fare','Embarked']
    usedCols_cont = ['Survived','Pclass','Sex','SibSp','Parch','Fare','Embarked']

    attributes_test = ['Fare']
    usedCols_test = ['Survived','Fare']



    simplifyBool = False
    includeContinous = True




    if includeContinous:
        filename = 'graph_continous.png'
        train_df = pd.read_csv(trainFile, usecols=usedCols_cont)
        test_df = pd.read_csv(testFile, usecols=usedCols_cont)


        print(f"\nStarting training on continous data set.")
        tree = decisionTreeLearning(train_df, train_df, attributes_cont, None, simplifyBool)

        tree_copy = tree.copy()
        graph = pydot.Dot(graph_type='graph')
        visit(tree_copy)
        graph.write_png(filename)

        print(f"\nFinished building the tree. Saved the image to '{filename}'. Simplifyed? {simplifyBool}")
        print('\nTesting the tree on our test data')

        accuracy = testTree(test_df, tree)
        print(f'Finished testing. Accuracy: {accuracy} %')

    else:
        filename = 'graph_categorical.png'
        print(f"\nStarting training on categorical data set.")
        train_df = pd.read_csv(trainFile, usecols=usedCols_disc)
        test_df = pd.read_csv(testFile, usecols=usedCols_disc)

        tree = decisionTreeLearning(train_df, train_df, attributes_disc, None, simplifyBool)

        tree_copy = tree.copy()
        graph = pydot.Dot(graph_type='graph')
        visit(tree_copy)
        print(graph.to_string())
        graph.write_png(filename)


        print(f"\nFinished building the tree. Saved the image to '{filename}'. Simplifyed? {simplifyBool}")
        print('\nTesting the tree on our test data')

        accuracy = testTree(test_df, tree)
        print(f'Finished testing. Accuracy: {accuracy} %')



    