import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot(dataFrameCompared, title):
	# Figure size
	fig, fig_real = plt.subplots(figsize=(10, 5))
	dataLength = len(dataFrameCompared.real)
	maxValue = max(dataFrameCompared.real)
	minValue = min(dataFrameCompared.real)

	lineReal = plt.plot(dataFrameCompared.index, dataFrameCompared.real, color='k', lw=2)[0]
	linePredict = plt.plot(dataFrameCompared.index, dataFrameCompared.predicted, color='r', lw=2)[0]

	def animate(i):
		lineReal.set_xdata(dataFrameCompared.index[:i])
		lineReal.set_ydata(dataFrameCompared.real[:i])

		linePredict.set_xdata(dataFrameCompared.index[:i])
		linePredict.set_ydata(dataFrameCompared.predicted[:i])
		

	anim = FuncAnimation(fig, animate, interval=200, frames=dataLength-1)
	plt.draw()
	plt.title(title)
	plt.legend()

	anim.save('animacao.mp4', fps=10, dpi=300)
	plt.show()
