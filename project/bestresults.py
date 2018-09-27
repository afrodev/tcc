# importação
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import csv

import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.metrics import mean_squared_error



def getDataset(filename='ITSA4-20140919-20180919-transformed.csv'):
	# Dataset com todos os valores do atual até -7
	originalDataset = pd.read_csv(filename, sep=';')
	originalDataset['Date'] = pd.to_datetime(originalDataset['Date'])
	originalDataset = originalDataset.set_index(['Date'])
	return originalDataset


def executeSVR(originalDataset, sizeDataset=550, trainPercent=0.7):

	originalDataset = originalDataset[:sizeDataset]

	# Atualiza apenas para valores com -7 e -6
	features = ['Open-7','Open-6','High-7','High-6', 'Low-7','Low-6', 'Close-7','Close-6', 'Close-0']
	transformedDataset = originalDataset[features]

	# Separação dos dados para treinamento e teste 70-30%
	lengthDataset = len(transformedDataset)
	trainLength = int(lengthDataset * trainPercent)

	trainDataset = transformedDataset[:trainLength]
	testDataset = transformedDataset[trainLength:lengthDataset]

	# Separando dados de entrada e de saída
	inputFeatures = ['Open-7','Open-6','High-7','High-6', 'Low-7','Low-6', 'Close-7','Close-6']
	outputFeatures = ['Close-0']
	x_input = trainDataset[inputFeatures]
	y_output = trainDataset[outputFeatures].values.ravel()

	# Treinando dados de entrada 
	svr_rbf = SVR(kernel='rbf')#, verbose=True)
	trainedModel = svr_rbf.fit(x_input, y_output)

	# Separando os dados de entrada e os de saída
	x_input_test = testDataset[inputFeatures]
	y_original_values = testDataset[outputFeatures].values.ravel()

	# Prediz valores do dataset de teste
	y_predicted_values = trainedModel.predict(x_input_test)

	# Criar um dataset novo com dados originais e os preditos
	data = {'date': testDataset.index, 'real': y_original_values, 'predicted': y_predicted_values}
	dataFrameCompared = pd.DataFrame(data=data)
	return dataFrameCompared


def plot(dataFrameCompared):
	# Gráficco
	fig = plt.gcf()
	fig.set_size_inches(18.5, 10.5, forward=True)

	plt.plot(dataFrameCompared.index, dataFrameCompared.real, color='navy', lw=2, label='REAL')
	plt.plot(dataFrameCompared.index, dataFrameCompared.predicted, color='green', lw=2, label='PREDITO')

	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()


# Mean Square Error
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Data Info to agregate all infos
class DataInfo:
	def __init__(self, sizeDataset, trainPercent, mse, mape):
		self.sizeDataset = sizeDataset
		self.trainPercent = trainPercent
		self.mse = mse
		self.mape = mape
		self.sizeTrainDataset = int(sizeDataset * trainPercent)
		self.sizeTestDataset = sizeDataset - self.sizeTrainDataset 



def csvFile(rowsInfo):
	with open('inforesults.csv', 'w', newline='') as csvfile:
	    spamwriter = csv.writer(csvfile, delimiter=';',
	                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)
	    
	    spamwriter.writerow(['SizeDataset', 'TrainPercent', 'sizeTrainDataset', 'sizeTestDataset', 'mse', 'mape'])
	    for r in rowsInfo:
	    	spamwriter.writerow([r.sizeDataset, r.trainPercent, r.sizeTrainDataset, r.sizeTestDataset, r.mse, r.mape])


def main():
	dataset = getDataset(filename='transformed-data/BBAS3.SA-20140910-20180922-transformed.csv')
	infos = []

	for sizeDataset in range(50, 1000):
		for trainPercent in np.arange(0.1, 0.9, 0.01):

			dataFrameCompared = executeSVR(originalDataset=dataset, sizeDataset=sizeDataset, trainPercent=trainPercent)
			mape = mean_absolute_percentage_error(dataFrameCompared.real, dataFrameCompared.predicted)
			mse = mean_squared_error(dataFrameCompared.real, dataFrameCompared.predicted) 
			
			print(str(sizeDataset) +';'+ str(trainPercent) + ';' + str(mse) + ';' + str(mape) + ';')
			infos.append(DataInfo(sizeDataset=sizeDataset, trainPercent=trainPercent, mse=mse, mape=mape))
			
	csvFile(infos)
	



# Main execution
# main()
