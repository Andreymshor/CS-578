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
	

	numRows = len(trainData)
	model = ID3(train_data, train_labels)
	#printTree(model)
	train_data_predicted = []
	for i in range(numRows):
		rowDict = buildRowDictionary(trainData.iloc[i]) # predicted train labels
		train_data_predicted.append(traverseTree(model, rowDict))

	validationData = readFile('validation.txt')
	testData = readFile('test.txt')
	original_test_labels = testData["A16"] # original test labels
	
	numRows = len(testData)
	test_data_predicted = []
	for i in range(numRows):
		rowDict = buildRowDictionary(testData.iloc[i])
		test_data_predicted.append(traverseTree(model, rowDict)) # predicted
	
	for i in range(numRows):
		rowDict = buildRowDictionary(validationData.iloc[i])
		test_data_predicted.append(traverseTree(model, rowDict)) # predicted
	return [list(train_labels), train_data_predicted, list(original_test_labels), test_data_predicted]




def DecisionTreeBounded(maxDepth):
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
	

	numRows = len(trainData)
	model = ID3(train_data, train_labels, 0, maxDepth)
	# printTree(model,1)
	train_data_predicted = []
	for i in range(numRows):
		rowDict = buildRowDictionary(trainData.iloc[i]) # predicted train labels
		train_data_predicted.append(traverseTree(model, rowDict))

	validationData = readFile('validation.txt')
	testData = readFile('test.txt')
	original_test_labels = testData["A16"] # original test labels
	
	numRows = len(testData)
	test_data_predicted = []
	for i in range(numRows):
		rowDict = buildRowDictionary(testData.iloc[i])
		test_data_predicted.append(traverseTree(model, rowDict)) # predicted

	# numRows = len(validationData)
	# validation_data_predicted = []
	# original_validation_labels = validationData["A16"]
	# for i in range(numRows):
	# 	rowDict = buildRowDictionary(validationData.iloc[i])
	# 	validation_data_predicted.append(traverseTree(model, rowDict)) # predicted

	# return [list(train_labels), train_data_predicted, list(original_validation_labels), validation_data_predicted]


	return [list(train_labels), train_data_predicted, list(original_test_labels), test_data_predicted]


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
	#print(before_split_freqs)
	gain = entropy(before_split_freqs)
	overall_size = sum(before_split_freqs)
	for freq in after_split_freqs:
		ratio = sum(freq) * 1.0 / overall_size
		gain -= ratio * entropy(freq)
	
	return gain


def generateThreshold(attr):
	attr = [float(x) for x in attr]
	attr.sort()
	thresholdList = []
	for i in range(0, len(attr) - 1):
		avg = (attr[i] + attr[i + 1]) / 2.0
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
	values = mergedData["A16"].value_counts().index.tolist()
	return values[0]

def buildRowDictionary(data):
	rowDict = dict()
	rowDict["A1"] = data[0]
	rowDict["A2"] = data[1]
	rowDict["A3"] = data[2]
	rowDict["A4"] = data[3]
	rowDict["A5"] = data[4]
	rowDict["A6"] = data[5]
	rowDict["A7"] = data[6]
	rowDict["A8"] = data[7]
	rowDict["A9"] = data[8]
	rowDict["A10"] = data[9]
	rowDict["A11"] = data[10]
	rowDict["A12"] = data[11]
	rowDict["A13"] = data[12]
	rowDict["A14"] = data[13]
	rowDict["A15"] = data[14]
	rowDict["A16"] = data[15]
	
	return rowDict

def calculateNumLabels(mergedData):
	pos = 0
	neg = 0
	for label in mergedData["A16"]:
		if label == 0:
			pos += 1
		else:
			neg += 1
	
	posNegList = [pos, neg]
	return posNegList


def optimalInfogainAndThreshold(mergedData, attr, thresholdList):
    beforeSplitList = calculateNumLabels(mergedData)
    infoGainList = []

    if len(thresholdList) == 0:
        return -1

    for threshold in thresholdList:
        posNegList = []
        df1 = mergedData[mergedData[attr] <= threshold]
        df2 = mergedData[mergedData[attr] > threshold]

        posNegList.append(calculateNumLabels(df1))
        posNegList.append(calculateNumLabels(df2))

        infoGainList.append([infor_gain(beforeSplitList, posNegList), threshold])

    maxGainThreshold = max(infoGainList, key=lambda x: x[0])

    return [maxGainThreshold[0], maxGainThreshold[1], attr]

def ID3(data, labels, currDepth=0, maxDepth=None):
	mergedData = mergeData(data, labels)  # stores the entire dataset along with train labels
	trainDict = generateDictionary(data)  # dictionary that stores threshold lists for each attribute

	if len(data.columns) == 0:
		leaf_node = Node(None, None, None, None, None, None, True, majorityValue(mergeData))
		return leaf_node
	
	if currDepth is not None and maxDepth is not None:
		if int(currDepth) >= int(maxDepth):
			leaf_node = Node(None, None, None, None, None, None, True, majorityValue(mergedData))
			return leaf_node
	
	# print(mergedData['A16'])
	if len(pd.unique(mergedData['A16'])) == 1:
		label = pd.unique(mergedData['A16'])
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
	rightMergedData = rightMergedData.drop(columns=the_chosen_attribute, axis=1)

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
		#print(currDepth)
		left_subtree = ID3(left_part_train_data, left_part_train_label, currDepth, maxDepth)
		
	else:
		#print(currDepth)
		left_subtree = ID3(left_part_train_data, left_part_train_label, currDepth)
		
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
		#print(currDepth)
		right_subtree = ID3(right_part_train_data, right_part_train_label, currDepth, maxDepth)
	else:
		#print(currDepth)
		right_subtree = ID3(right_part_train_data, right_part_train_label, currDepth)
	
	current_node.left_subtree = left_subtree
	current_node.right_subtree = right_subtree
	return current_node 


def traverseTree(node, rowDict):
	label = 0
	if node.isLeafNode:
		return node.label
	
	#print(rowDict[node.attribute])
	if rowDict[node.attribute] <= node.threshold:
		if node.left_subtree is not None:
			label = traverseTree(node.left_subtree, rowDict)
	
	if rowDict[node.attribute] > node.threshold:
		if node.right_subtree is not None:
			label = traverseTree(node.right_subtree, rowDict)
	
	return label


def printTree(root, d):
	print('-'*d, d)
	if root.left_subtree is None and root.right_subtree is None:
		print(f"leaf label: {root.label}")
	if root.left_subtree is not None and root.right_subtree is not None:
		print(f"Root Attribute: {root.attribute}")
		print(f"Root threshold: {root.threshold}")
		print(f"Root InfoGain: {root.infoGain}")
		printTree(root.left_subtree, d+1)
		printTree(root.right_subtree, d+1)
	if root.left_subtree is not None and root.right_subtree is None:
		print(f"Root Attribute: {root.attribute}")
		print(f"Root Threshold: {root.threshold}")
		print(f"Root InfoGain: {root.infoGain}")
		printTree(root.left_subtree, d+1)

	if root.right_subtree is not None and root.left_subtree is None:
		print(f"Root Attribute: {root.attribute}")
		print(f"Root Threshold: {root.threshold}")
		print(f"Root InfoGain: {root.infoGain}")
		printTree(root.right_subtree, d+1)


# reads and cleans the file
def readFile(filename):
	file = pd.read_csv(filename, sep='\t', header=None)
	columns = list(file)
	cat_columns = [0,3,4,5,6,8,9,11,12,15] # positive is 0, negative is 1
	for col in columns:
		if col in cat_columns:
			file[col] = file[col].replace(np.nan, file[col].mode()[0])
			file[col] = file[col].replace('?', file[col].mode()[0])
			_, numerical = np.unique(file[col].to_numpy(), return_inverse=True)
			file[col] = numerical
		
		#print(col)
		else:
			continuousData = [float(x) for x in file[col] if x != '?']
			meanData = sum(continuousData) / len(continuousData)
			file[col] = file[col].replace(np.nan, meanData)
			file[col] = file[col].replace('?', meanData)
		
		file[col] = [float(x) for x in file[col]]
	file.columns = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"]
	
	return file




def main():
	# trainData = readFile('train.txt')
	# validationData = readFile('validation.txt')
	# testData = readFile('test.txt')

	print(DecisionTree())
	#print(trainData)

if __name__=='__main__':
	main()