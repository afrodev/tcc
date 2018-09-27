import files
import preprocessing as ppc
import machinelearning as ml
import plotter as pl
import error as er
import numpy as np
import windowing as wdw
import results as rs

def main():
	daysAhead = 1
	sizeDataset = 550
	trainPercent = 0.7

	originalDataset = files.getDataset(filename='../original-data/' + 'ITSA4-20140919-20180919.csv')
	preprocessedDataset = ppc.preprocessing(daysAhead=daysAhead, dataset=originalDataset)

	rs.getResultsFixedWindow(stockName='itsa4', dataset=preprocessedDataset, daysAhead=1, 
		rangeSizeDataset=range(550,551), rangeTrainPercent=np.arange(0.7, 0.71, 0.01))

	rs.getResultsSlidingWindow(stockName='itsa4', dataset=preprocessedDataset, daysAhead=1, 
		rangeSizeDataset=range(550,551), rangeTrainPercent=np.arange(0.7, 0.71, 0.01))

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