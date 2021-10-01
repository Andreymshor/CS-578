import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
eps = np.finfo(float).eps
from numpy import log2 as log




def DecisionTree():
		#TODO: Your code starts from here. 
        #      This function should return a list of labels.
	#      e.g.: 
	#	labels = [['+','-','+'],['+','+','-'],['-','+'],['+','+']]
       	#	return labels
	#	where:
	#		labels[0] = original_training_labels
	#		labels[1] = prediected_training_labels
	#		labels[2] = original_testing_labels
	#		labels[3] = predicted_testing_labels


	trainData = readFile('train.txt')
	train_labels = trainData["A16"] # original train labels
	train_data = trainData.drop('A16', axis=1) # original train data

	print(train_data.columns)

	model = ID3(trainData, train_labels)
	rowDict = buildRowDictionary(trainData) # predicted train labels
	train_data_predicted = traverseTree(model, rowDict)

	validationData = readFile('validation.txt')
	testData = readFile('test.txt')
	original_test_labels = testData["A16"] # original test labels
	rowDict = buildRowDictionary(testData)
	test_data_predicted = traverseTree(model, rowDict) # predicted

	return [train_labels, train_data_predicted, original_test_labels, test_data_predicted]




def DecisionTreeBounded(data, maxDepth):
		#TODO: Your code starts from here.
    #      This function should return a list of labels.
	#      e.g.:
	#	labels = [['+','-','+'],['+','+','-'],['-','+'],['+','+']]
    #	return labels
	#	where:
	#		labels[0] = original_training_labels
	#		labels[1] = prediected_training_labels
	#		labels[2] = original_testing_labels
	#		labels[3] = predicted_testing_labels
    
	return

class Node(object):
    def __init__(self, l=None, r=None, attr=None, thresh=None, infogain = None, majorityVote = None, isLeafNode=False, label=None):
        self.left_subtree = l
        self.right_subtree = r
        self.attribute = attr
        self.threshold = thresh
        self.isLeafNode = isLeafNode
        self.infoGain = infogain
        self.majorityVote = majorityVote
        if self.isLeafNode:
            self.label = label

def entropy(freqs):
	all_freq = sum(freqs)
	entropy = 0
	for fq in freqs:
		prob = fq * 1.0 / all_freq
		if abs(prob) > 1e-8:
			entropy += -prob * np.log2(prob)
	
	return entropy

def infor_gain(before_split_freqs, after_split_freqs):
    gain = entropy(before_split_freqs)
    overall_size = sum(before_split_freqs)
    for freq in after_split_freqs:
        ratio = sum(freq) * 1.0 / overall_size
        gain -= ratio * entropy(freq)
    return gain



def generateThreshold(attr):
    attr.sort()
    thresholdList = []
    for i in range(0, len(attr) - 1):
        avg = (attr[i] + attr[i + 1]) * 1.0 / 2.0
        if min(attr) < avg < max(attr):
            thresholdList.append(avg)

    return thresholdList

def generateDictionary(trainData):
    # print(trainData.to_string())
    thresholdDict = dict()
    for col in trainData:
        if not (col in thresholdDict):
            thresholdDict[col] = generateThreshold(list(trainData[col]))
    return thresholdDict

def mergeData(trainData, trainLabel):
    return pd.concat([trainData, trainLabel], axis=1)

def majorityValue(mergedData):
	values = mergedData[15].value_counts().index.tolist()
	return values[0]

def buildRowDictionary(data):
	rowDict = dict()
	columns = list(data)
	for col in columns:
		rowDict[str(col)] = data[col]
	
	return rowDict

def calculateNumLabels(mergedData):
	pos = 0
	neg = 0
	for label in mergeData[15]:
		if label == 0:
			pos += 1
		else:
			neg += 1
	
	posNegList = [pos, neg]
	return [posNegList]


def optimalInfogainAndThreshold(mergedData, attr, thresholdList):
    beforeSplitList = calculateNumLabels(mergedData)
    infoGainList = []

    if len(thresholdList) == 0:
        return -1

    for threshold in thresholdList:
        survivedOrDeadList = []
        df1 = mergedData[mergedData[attr] <= threshold]
        df2 = mergedData[mergedData[attr] > threshold]

        survivedOrDeadList.append(calculateNumLabels(df1))
        survivedOrDeadList.append(calculateNumLabels(df2))

        infoGainList.append([infor_gain(beforeSplitList, survivedOrDeadList), threshold])

    maxGainThreshold = max(infoGainList, key=lambda x: x[0])

    return [maxGainThreshold[0], maxGainThreshold[1], attr]

def ID3(data, labels, currDepth=None, maxDepth=None):
	mergedData = mergeData(data, labels)  # stores the entire dataset along with train labels
	trainDict = generateDictionary(data)  # dictionary that stores threshold lists for each attribute

	if len(data.columns) == 0:
		leaf_node = Node(None, None, None, None, None, None, True, majorityValue(mergeData))
		return leaf_node
	
	if currDepth is not None and maxDepth is not None:
		if int(currDepth) >= int(maxDepth):
			leaf_node = Node(None, None, None, None, None, None, True, majorityValue(mergedData))
			return leaf_node
	
	if len(pd.unique(mergedData['A16'])) == 1:
		label = pd.unique(mergeData['A16'])
		leaf_node = Node(None, None, None, None, None, None, True, label[0])
		return leaf_node
	

	# 1. use a for loop to calculate the infor-gain of every attribute
	optimalInfoGainList = [] # will store the optimal thresholds for each attribute and information gain
	noSplitList = []
	for attr in data:
		thresholdList = trainDict[attr]
		if optimalInfogainAndThreshold(mergedData, attr, thresholdList) == -1:
			noSplitList.append(attr)
		else:
			optimalInfoGainList.append(optimalInfogainAndThreshold(mergedData, attr, thresholdList))
		
	if len(noSplitList) == len(data):
		leaf_node = Node(None, None, None, None, None, None, True, majorityValue(mergedData))
		return leaf_node
	
	# 2. pick the attribute that achieve the maximum infor-gain
    # 0th attribute = info_gain, 1st attribute = threshold, 2nd attribute = attr

	if len(optimalInfoGainList) == 0:
		leaf_node = Node(None, None, None, None, None, None, True, majorityValue(mergedData))
		return leaf_node
	
	bestAttr = max(optimalInfoGainList, key=lambda x: x[0])
	the_chosen_attribute = bestAttr[2]
	the_chosen_threshold = bestAttr[1]
	the_chosen_infogain = bestAttr[0]

	# 3. build a node to hold the data;
	current_node = Node(None, None, the_chosen_attribute, the_chosen_threshold, the_chosen_infogain, majorityValue(mergedData))

	# 4. Split data into two parts

	leftMergedData = mergedData[mergedData[the_chosen_attribute] <= the_chosen_threshold]
	rightMergedData = mergedData[mergedData[the_chosen_attribute] > the_chosen_threshold]

	# drop the column
	leftMergedData = leftMergedData.drop(columns=the_chosen_attribute, axis=1)
	rightMergedData = rightMergedData.drop(columns=the_chosen_attribute, axis=2)

	listOfRemainingColumns = []

	for col in leftMergedData.columns:
		listOfRemainingColumns.append(col)
	
	# 5. call ID3() for the left parts of the data
	left_part_train_data = leftMergedData[listOfRemainingColumns]
	left_part_train_label = leftMergedData["A16"]
	left_part_train_data = left_part_train_data.drop(columns=["A16"], axis=1)
	
	left_subtree = None
	if currDepth is not None and maxDepth is not None:
		currDepth += 1
		left_subtree = ID3(left_part_train_data, left_part_train_label, currDepth, maxDepth)

	else:
		left_subtree = ID3(left_part_train_data, left_part_train_label)
	
	# 6. call ID3 for the right part of the data

	listOfRemainingColumns = []
	for col in rightMergedData.columns:
		listOfRemainingColumns.append(col)
	
	right_part_train_data = rightMergedData[listOfRemainingColumns]
	right_part_train_label = rightMergedData['A16']
	right_part_train_data = right_part_train_data.drop(columns=['A16'], axis=1)

	right_subtree = None
	if currDepth is not None and maxDepth is not None:
		currDepth += 1
		right_subtree = ID3(right_part_train_data, left_part_train_label, currDepth, maxDepth)
	else:
		right_subtree = ID3(right_part_train_data, right_part_train_label)
	
	current_node.left_subtree = left_subtree
	current_node.right_subtree = right_subtree
	return current_node 



def traverseTree(node, rowDict):
    label = 0
    if node.isLeafNode:  # Base case
        return node.label

    if rowDict[node.attribute] <= node.threshold:
        if node.left_subtree is not None:
            label = traverseTree(node.left_subtree, rowDict)

    if rowDict[node.attribute] > node.threshold:
        if node.right_subtree is not None:
            label = traverseTree(node.right_subtree, rowDict)

    return label





# reads and cleans the file
def readFile(filename):
	file = pd.read_csv(filename, sep='\t', header=None)
	columns = list(file)
	cat_columns = [0,3,4,5,6,8,9,11,12,15] # positive is 0, negative is 1
	for col in columns:
		values = file[col].value_counts().index.tolist()

		file[col] = file[col].replace(np.nan, values[0])
		if col in cat_columns:
			currCol = file[col].to_numpy()
			_, numerical = np.unique(currCol, return_inverse=True)
			file[col] = numerical
	file.columns = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"]
	
	return file




def main():
	# trainData = readFile('train.txt')
	# validationData = readFile('validation.txt')
	# testData = readFile('test.txt')

	DecisionTree()
	#print(trainData)

if __name__=='__main__':
	main()