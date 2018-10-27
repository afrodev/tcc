import files
import preprocessing as ppc
import machinelearning as ml
import error as er
from sklearn.metrics import mean_squared_error

def fixedWindow(dataset, daysAhead=1, sizeDataset=550, trainPercent=0.7):
	dataFrameCompared = ml.executeSVRForDaysAhead(isSlidingWindow=False, daysAhead=daysAhead, dataset=dataset, sizeDataset=sizeDataset, trainPercent=trainPercent)
	mape = er.mean_absolute_percentage_error(dataFrameCompared.real, dataFrameCompared.predicted)
	mse = mean_squared_error(dataFrameCompared.real, dataFrameCompared.predicted) 

	print(str(sizeDataset) +';'+ str(trainPercent) + ';' + str(mse) + ';' + str(mape) + ';', end='\r')

	return files.DataInfo(sizeDataset=sizeDataset, trainPercent=trainPercent, mse=mse, mape=mape, dataFrameCompared=dataFrameCompared)

def slidingWindow(dataset, daysAhead=1, sizeDataset=550, trainPercent=0.7):
	dataFrameCompared = ml.executeSVRForDaysAhead(isSlidingWindow=True, daysAhead=daysAhead, dataset=dataset, sizeDataset=sizeDataset, trainPercent=trainPercent)
	mape = er.mean_absolute_percentage_error(dataFrameCompared.real, dataFrameCompared.predicted)
	mse = mean_squared_error(dataFrameCompared.real, dataFrameCompared.predicted) 
	#pl.plot(dataFrameCompared=dataFrameCompared)
	print(str(sizeDataset) +';'+ str(trainPercent) + ';' + str(mse) + ';' + str(mape) + ';', end='\r')

	return files.DataInfo(sizeDataset=sizeDataset, trainPercent=trainPercent, mse=mse, mape=mape, dataFrameCompared=dataFrameCompared)