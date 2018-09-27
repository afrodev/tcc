import csv
import pandas as pd


def getDataset(filename):
	# Dataset com todos os valores do atual até -7
	originalDataset = pd.read_csv(filename, sep=',')
	originalDataset['Date'] = pd.to_datetime(originalDataset['Date'])
	originalDataset = originalDataset.set_index(['Date'])
	return originalDataset


# Data Info to agregate all infos
class DataInfo:
	def __init__(self, sizeDataset, trainPercent, mse, mape, dataFrameCompared):
		self.sizeDataset = sizeDataset
		self.trainPercent = trainPercent
		self.mse = mse
		self.mape = mape
		self.sizeTrainDataset = int(sizeDataset * trainPercent)
		self.sizeTestDataset = sizeDataset - self.sizeTrainDataset 
		self.dataFrameCompared = dataFrameCompared


def csvFile(name, rowsInfo):
	filename = './results/results-' + str(name) + '.csv'
	with open(filename, 'w', newline='') as csvfile:
	    spamwriter = csv.writer(csvfile, delimiter=';',
	                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)
	    
	    spamwriter.writerow(['SizeDataset', 'TrainPercent', 'sizeTrainDataset', 'sizeTestDataset', 'mse', 'mape'])
	    for r in rowsInfo:
	    	spamwriter.writerow([r.sizeDataset, r.trainPercent, r.sizeTrainDataset, r.sizeTestDataset, r.mse, r.mape])
