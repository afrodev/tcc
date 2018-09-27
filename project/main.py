import files
import preprocessing as ppc
import machinelearning as ml
import plotter as pl
import error as er
import numpy as np
from sklearn.metrics import mean_squared_error


def results(stockName, filename, daysAhead, rangeSizeDataset, rangeTrainPercent):
	originalDataset = files.getDataset(filename='../original-data/' + filename)
	preprocessedDataset = ppc.preprocessing(daysAhead=daysAhead, dataset=originalDataset)

	infos = []
	for sizeDataset in rangeSizeDataset:
		for trainPercent in rangeTrainPercent:
			dataFrameCompared = ml.executeSVRForDaysAhead(daysAhead=daysAhead, dataset=preprocessedDataset, sizeDataset=sizeDataset, trainPercent=trainPercent)
			mape = er.mean_absolute_percentage_error(dataFrameCompared.real, dataFrameCompared.predicted)
			mse = mean_squared_error(dataFrameCompared.real, dataFrameCompared.predicted) 
			print(str(sizeDataset) +';'+ str(trainPercent) + ';' + str(mse) + ';' + str(mape) + ';')
			infos.append(files.DataInfo(sizeDataset=sizeDataset, trainPercent=trainPercent, mse=mse, mape=mape))

	files.csvFile(name=stockName, rowsInfo=infos)



def main():
	results(stockName='itsa4', filename='ITSA4-20140919-20180919.csv', daysAhead=1, 
		rangeSizeDataset=range(50,55), rangeTrainPercent=np.arange(0.1, 0.9, 0.01))

	#pl.plot(dataFrameCompared=dataFrameCompared)

main()
