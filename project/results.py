import files
import windowing as wdw

def getResultsSlidingWindow(stockName, dataset, daysAhead, rangeSizeDataset, rangeTrainPercent):
	infos = []

	for sizeDataset in rangeSizeDataset:
		for trainPercent in rangeTrainPercent:
			i = wdw.slidingWindow(dataset=dataset, daysAhead=daysAhead, sizeDataset=sizeDataset, trainPercent=trainPercent)
			infos.append(i)

	files.csvFile(name=stockName + '-slidingWindow-' + str(daysAhead) + '-daysAhead', rowsInfo=infos)

def getResultsFixedWindow(stockName, dataset, daysAhead, rangeSizeDataset, rangeTrainPercent):
	infos = []

	for sizeDataset in rangeSizeDataset:
		for trainPercent in rangeTrainPercent:
			i = wdw.slidingWindow(dataset=dataset, daysAhead=daysAhead, sizeDataset=sizeDataset, trainPercent=trainPercent)
			infos.append(i)
			
	files.csvFile(name=stockName + '-fixedWindow-' + str(daysAhead) + '-daysAhead', rowsInfo=infos)