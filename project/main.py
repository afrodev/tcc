import files
import preprocessing as ppc
import machinelearning as ml
import plotter as pl
import error as er
import numpy as np
from sklearn.metrics import mean_squared_error


# def results(stockName, filename, daysAhead, rangeSizeDataset, rangeTrainPercent):
# 	originalDataset = files.getDataset(filename='../original-data/' + filename)
# 	preprocessedDataset = ppc.preprocessing(daysAhead=daysAhead, dataset=originalDataset)

# 	infos = []

# 	for sizeDataset in rangeSizeDataset:
# 		for trainPercent in rangeTrainPercent:
# 			dataFrameCompared = ml.executeSVRForDaysAhead(daysAhead=daysAhead, dataset=preprocessedDataset, sizeDataset=sizeDataset, trainPercent=trainPercent)
# 			mape = er.mean_absolute_percentage_error(dataFrameCompared.real, dataFrameCompared.predicted)
# 			mse = mean_squared_error(dataFrameCompared.real, dataFrameCompared.predicted) 
# 			print(str(sizeDataset) +';'+ str(trainPercent) + ';' + str(mse) + ';' + str(mape) + ';')
# 			infos.append(files.DataInfo(sizeDataset=sizeDataset, trainPercent=trainPercent, mse=mse, mape=mape))

# 	files.csvFile(name=stockName, rowsInfo=infos)

def fixedWindow(dataset, daysAhead=1, sizeDataset=550, trainPercent=0.7):
	dataFrameCompared = ml.executeSVRForDaysAhead(isSlidingWindow=False, daysAhead=daysAhead, dataset=dataset, sizeDataset=sizeDataset, trainPercent=trainPercent)
	mape = er.mean_absolute_percentage_error(dataFrameCompared.real, dataFrameCompared.predicted)
	mse = mean_squared_error(dataFrameCompared.real, dataFrameCompared.predicted) 
	print(str(sizeDataset) +';'+ str(trainPercent) + ';' + str(mse) + ';' + str(mape) + ';')

	return files.DataInfo(sizeDataset=sizeDataset, trainPercent=trainPercent, mse=mse, mape=mape)

def slidingWindow(dataset, daysAhead=1, sizeDataset=550, trainPercent=0.7):
	dataFrameCompared = ml.executeSVRForDaysAhead(isSlidingWindow=True, daysAhead=daysAhead, dataset=dataset, sizeDataset=sizeDataset, trainPercent=trainPercent)
	mape = er.mean_absolute_percentage_error(dataFrameCompared.real, dataFrameCompared.predicted)
	mse = mean_squared_error(dataFrameCompared.real, dataFrameCompared.predicted) 
	#pl.plot(dataFrameCompared=dataFrameCompared)
	print(str(sizeDataset) +';'+ str(trainPercent) + ';' + str(mse) + ';' + str(mape) + ';')

	return files.DataInfo(sizeDataset=sizeDataset, trainPercent=trainPercent, mse=mse, mape=mape)


def main():
	daysAhead = 1
	sizeDataset = 550
	trainPercent = 0.7

	originalDataset = files.getDataset(filename='../original-data/' + 'ITSA4-20140919-20180919.csv')
	preprocessedDataset = ppc.preprocessing(daysAhead=daysAhead, dataset=originalDataset)

	slidingWindow(dataset=preprocessedDataset, daysAhead=daysAhead, sizeDataset=sizeDataset, trainPercent=trainPercent)
	fixedWindow(dataset=preprocessedDataset, daysAhead=daysAhead, sizeDataset=sizeDataset, trainPercent=trainPercent)

	#results(stockName='itsa4', filename='ITSA4-20140919-20180919.csv', daysAhead=1, 
	#	rangeSizeDataset=range(550,551), rangeTrainPercent=np.arange(0.7, 0.71, 0.01))

	#pl.plot(dataFrameCompared=dataFrameCompared)

main()
