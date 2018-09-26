from sklearn.svm import SVR
import pandas as pd

features1dayAhead = ['open-2', 'high-2', 'low-2', 'close-2','close-0']
features5daysAhead = ['open-7','open-6','high-7','high-6', 'low-7','low-6', 'close-7','close-6', 'close-0']
features22daysAhead = ['open-24','open-23','high-24','high-23', 'low-24','low-23', 'close-24','close-23', 'close-0']

def executeSVRForDaysAhead(daysAhead, dataset, sizeDataset, trainPercent):
	if daysAhead == 1:
		return executeSVR(features=features1dayAhead, originalDataset=dataset, sizeDataset=sizeDataset, trainPercent=trainPercent)
	elif daysAhead == 5:
		return executeSVR(features=features5daysAhead, originalDataset=dataset, sizeDataset=sizeDataset, trainPercent=trainPercent)
	elif daysAhead == 22:
		return executeSVR(features=features22daysAhead, originalDataset=dataset, sizeDataset=sizeDataset, trainPercent=trainPercent)
	else:
		print("invalid svr date")

def executeSVR(features, originalDataset, sizeDataset=550, trainPercent=0.7):

	originalDataset = originalDataset[:sizeDataset]
	transformedDataset = originalDataset[features]

	# Separação dos dados para treinamento e teste 70-30%
	lengthDataset = len(transformedDataset)
	trainLength = int(lengthDataset * trainPercent)

	trainDataset = transformedDataset[:trainLength]
	testDataset = transformedDataset[trainLength:lengthDataset]

	# Separando dados de entrada e de saída
	inputFeatures = features[:-1] # Get features from the first to one before the last
	outputFeatures = features[-1] # Get the last one on array
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