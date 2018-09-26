import files
import preprocessing as ppc
import machinelearning as ml
import plotter as pl
import error as er

import numpy as np
from sklearn.metrics import mean_squared_error


def main():
	originalDataset = files.getDataset(filename='original-data/ITSA4-20140919-20180919.csv')
	print("FINISH GET ORIGINAL DATASET")

	preprocessedDataset = ppc.preprocessing(daysAhead=1, dataset=originalDataset)
	print("PREPROCESSING DATA")
	
	infos = []

	for sizeDataset in range(50, 55):
		for trainPercent in np.arange(0.1, 0.9, 0.01):
			dataFrameCompared = ml.executeSVRForDaysAhead(daysAhead=1, dataset=preprocessedDataset, sizeDataset=sizeDataset, trainPercent=trainPercent)

			mape = er.mean_absolute_percentage_error(dataFrameCompared.real, dataFrameCompared.predicted)
			mse = mean_squared_error(dataFrameCompared.real, dataFrameCompared.predicted) 
			
			print(str(sizeDataset) +';'+ str(trainPercent) + ';' + str(mse) + ';' + str(mape) + ';')
			infos.append(files.DataInfo(sizeDataset=sizeDataset, trainPercent=trainPercent, mse=mse, mape=mape))
			

	print("dataframeCompared")

	print(dataFrameCompared)
	print("PRINCIPAL")

	pl.plot(dataFrameCompared=dataFrameCompared)

main()


# def bestResult():
# 	dataset = getDataset(filename='original-data/BBAS3.SA-20140910-20180922-transformed.csv')
# 	infos = []

# 	for sizeDataset in range(50, 1000):
# 		for trainPercent in np.arange(0.1, 0.9, 0.01):

# 			dataFrameCompared = executeSVR(originalDataset=dataset, sizeDataset=sizeDataset, trainPercent=trainPercent)
# 			mape = mean_absolute_percentage_error(dataFrameCompared.real, dataFrameCompared.predicted)
# 			mse = er.mean_squared_error(dataFrameCompared.real, dataFrameCompared.predicted) 
			
# 			print(str(sizeDataset) +';'+ str(trainPercent) + ';' + str(mse) + ';' + str(mape) + ';')
# 			infos.append(DataInfo(sizeDataset=sizeDataset, trainPercent=trainPercent, mse=mse, mape=mape))
			
# 	files.csvFile(infos)

# #bestResult()	
