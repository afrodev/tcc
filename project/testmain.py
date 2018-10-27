import files
import preprocessing as ppc
import machinelearning as ml
import plotter as pl
import error as er
import numpy as np
import windowing as wdw
import results as rs
from sklearn.svm import SVR
import pandas as pd
import pymp


features1dayAhead = ['open-2', 'high-2', 'low-2', 'close-2','close-0']
features5daysAhead = ['open-7','open-6','high-7','high-6', 'low-7','low-6', 'close-7','close-6', 'close-0']
features22daysAhead = ['open-24','open-23','high-24','high-23', 'low-24','low-23', 'close-24','close-23', 'close-0']

def slWithThreading(features, originalDataset, sizeDataset=550, trainPercent=0.7):

	nProcessadores = 4

	originalDataset = originalDataset[:sizeDataset]
	transformedDataset = originalDataset[features]

	lengthDataset = len(transformedDataset)
	trainLength = int(lengthDataset * trainPercent)
	testLength = lengthDataset - trainLength 

	# Separando dados de entrada e de saída
	inputFeatures = features[:-1] # Get features from the first to one before the last
	outputFeatures = features[-1] # Get the last one on array

	#y_original_values = pymp.shared.array((1, ), dtype='uint32')
	#y_predicted_values = pymp.shared.array((1, ), dtype='uint32')
	y_original_values = []
	y_predicted_values = []

	# Train dataset initial + 1 finish + 1
	# test datase initial + 1 finish = length

	with pymp.Parallel(nProcessadores) as p:
		for offset in range(0, testLength):
			#print('offset-executing: ' + str(offset))
			trainDataset = transformedDataset[offset:trainLength+offset]
			testDataset = transformedDataset[offset+trainLength:lengthDataset]

			x_input = trainDataset[inputFeatures]
			y_output = trainDataset[outputFeatures].values.ravel()

			# Treinando dados de entrada 
			svr_rbf = SVR(kernel='rbf')#, verbose=True)
			trainedModel = svr_rbf.fit(x_input, y_output)

			# Separando os dados de entrada e os de saída
			x_input_test = testDataset[inputFeatures]
			y_original_value = testDataset[outputFeatures].values.ravel()[0]

			# Prediz valores do dataset de teste
			y_predicted_value = trainedModel.predict(x_input_test)[0]
			#print('trainSize go from: ' + str(offset) + ' to ' + str(trainLength+offset))
			#print('testSize go from: ' + str(offset+trainLength) + ' to ' + str(lengthDataset))
			#print('offset: ' + str(offset) + ' --- predict:' + str(y_predicted_value) + ' --- original:' + str(y_original_value))
			
			y_original_values.append(y_original_value)
			y_predicted_values.append(y_predicted_value)



		testDataset = transformedDataset[trainLength:lengthDataset]
		data = {'date': testDataset.index, 'real': y_original_values, 'predicted': y_predicted_values}
		dataFrameCompared = pd.DataFrame(data=data)

		return dataFrameCompared

def oneResultThread():

	daysAhead = 1
	sizeDataset = 550
	trainPercent = 0.7

	originalDataset = files.getDataset(filename='../original-data/' + 'ITSA4-20140919-20180919.csv')
	preprocessedDataset = ppc.preprocessing(daysAhead=daysAhead, dataset=originalDataset)
	
	dataFrameCompared = slWithThreading(features=features1dayAhead, originalDataset=preprocessedDataset, sizeDataset=sizeDataset, trainPercent=trainPercent)
	#pl.plot(dataFrameCompared=dataFrameCompared, title='REAL vs Predito')

def oneResult():
	daysAhead = 1
	sizeDataset = 550
	trainPercent = 0.7

	originalDataset = files.getDataset(filename='../original-data/' + 'ITSA4-20140919-20180919.csv')
	preprocessedDataset = ppc.preprocessing(daysAhead=daysAhead, dataset=originalDataset)
	
	dataInfoSlidingWindow = wdw.slidingWindow(dataset=preprocessedDataset, daysAhead=daysAhead, sizeDataset=sizeDataset, trainPercent=trainPercent)
	#pl.plot(dataFrameCompared=dataInfoSlidingWindow.dataFrameCompared, title='REAL vs Predito')


import time

print('---------ORIGINAL-----------')
start_time = time.time()
oneResult()
print("--- %s seconds ---" % (time.time() - start_time))


print('---------THREAD-----------')
start_time = time.time()
oneResultThread()
print("--- %s seconds ---" % (time.time() - start_time))

