import matplotlib.pyplot as plt

def plot(dataFrameCompared):
	# Gráficco
	fig = plt.gcf()
	fig.set_size_inches(18.5, 10.5, forward=True)

	plt.plot(dataFrameCompared.index, dataFrameCompared.real, color='navy', lw=2, label='REAL')
	plt.plot(dataFrameCompared.index, dataFrameCompared.predicted, color='green', lw=2, label='PREDITO')

	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()