import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
eps = np.finfo(float).eps
from numpy import log2 as log




def DecisionTree(data):
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



def DecisionTree(maxDepth):
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
	pass




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

	
	return file




def main():
	trainData = readFile('train.txt')
	validationData = readFile('validation.txt')
	testData = readFile('test.txt')
	print(trainData[15])

if __name__=='__main__':
	main()