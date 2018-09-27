import files
import preprocessing as ppc
import machinelearning as ml
import plotter as pl
import error as er
import numpy as np
import windowing as wdw
import results as rs

def results():
	originalDataset = files.getDataset(filename='../original-data/' + 'ITSA4-20140919-20180919.csv')

	preprocessedDataset1Day = ppc.preprocessing(daysAhead=1, dataset=originalDataset)
	preprocessedDataset5Days = ppc.preprocessing(daysAhead=5, dataset=originalDataset)
	preprocessedDataset22Days = ppc.preprocessing(daysAhead=22, dataset=originalDataset)

	print('------ FIXED WINDOW 1 DAY -------')
	rs.getResultsFixedWindow(stockName='itsa4', dataset=preprocessedDataset1Day, daysAhead=1, 
		rangeSizeDataset=range(50,1000), rangeTrainPercent=np.arange(0.5, 0.90, 0.01))

	print('------ FIXED WINDOW 5 DAYS -------')
	rs.getResultsFixedWindow(stockName='itsa4', dataset=preprocessedDataset5Days, daysAhead=5, 
		rangeSizeDataset=range(50,1000), rangeTrainPercent=np.arange(0.5, 0.90, 0.01))

	print('------ SLIDING WINDOW 1 DAY -------')
	rs.getResultsSlidingWindow(stockName='itsa4', dataset=preprocessedDataset1Day, daysAhead=1, 
		rangeSizeDataset=range(50,1000), rangeTrainPercent=np.arange(0.5, 0.90, 0.01))

	print('------ SLIDING WINDOW 5 DAYS -------')
	rs.getResultsSlidingWindow(stockName='itsa4', dataset=preprocessedDataset5Days, daysAhead=5, 
		rangeSizeDataset=range(50,1000), rangeTrainPercent=np.arange(0.5, 0.90, 0.01))

def main():
	daysAhead = 1
	sizeDataset = 550
	trainPercent = 0.7

	originalDataset = files.getDataset(filename='../original-data/' + 'ITSA4-20140919-20180919.csv')
	preprocessedDataset = ppc.preprocessing(daysAhead=daysAhead, dataset=originalDataset)
	
	dataInfoSlidingWindow = wdw.slidingWindow(dataset=preprocessedDataset, daysAhead=daysAhead, sizeDataset=sizeDataset, trainPercent=trainPercent)
	#dataInfoFixedWindow = wdw.fixedWindow(dataset=preprocessedDataset, daysAhead=daysAhead, sizeDataset=sizeDataset, trainPercent=trainPercent)

	#pl.plot(dataFrameCompared=dataInfoSlidingWindow.dataFrameCompared)
	#pl.plot(dataFrameCompared=dataInfoFixedWindow.dataFrameCompared)
	pl.plot(dataFrameCompared=dataInfoSlidingWindow.dataFrameCompared, title='REAL vs Predito')

main()


'''
some examples:
	#wdw.slidingWindow(dataset=preprocessedDataset, daysAhead=daysAhead, sizeDataset=sizeDataset, trainPercent=trainPercent)
	#wdw.fixedWindow(dataset=preprocessedDataset, daysAhead=daysAhead, sizeDataset=sizeDataset, trainPercent=trainPercent)

	rs.getResultsFixedWindow(stockName='itsa4', dataset=preprocessedDataset, daysAhead=1, 
		rangeSizeDataset=range(550,551), rangeTrainPercent=np.arange(0.7, 0.71, 0.01))

	rs.getResultsSlidingWindow(stockName='itsa4', dataset=preprocessedDataset, daysAhead=1, 
		rangeSizeDataset=range(550,551), rangeTrainPercent=np.arange(0.7, 0.71, 0.01))

	pl.plot(dataFrameCompared=dataFrameCompared)s
'''