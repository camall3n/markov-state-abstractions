from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy
import numpy as np

def xsinx(x):
	"given a vector x, output the xsinx function"
	sin=numpy.mean(numpy.sin(x))
	norm=numpy.linalg.norm(x)
	return norm*sin

def x2(x):
	return numpy.linalg.norm(x)

def xsinx_function_get_batch(batch_size=256,num_dims=2):
	"given a batch size and dimension size, return a batch of x,y sampled from ackley"
	x_batch=numpy.random.uniform(low=-5, high=5, size=num_dims*batch_size)
	x_batch=x_batch.reshape(batch_size,num_dims)
	y_batch=[xsinx(x) for x in x_batch]
	y_batch=numpy.array(y_batch).reshape(batch_size,1)
	return x_batch,y_batch


if __name__=="__main__":
	X = np.arange(-5, 5, 0.5)
	Y = np.arange(-5, 5, 0.5)
	X, Y = np.meshgrid(X, Y)
	Z=numpy.zeros_like(X)

	for x_index in range(len(X)):
		for y_index in range(len(X)):
			arr=numpy.array([X[x_index,y_index],Y[x_index,y_index]])
			Z[x_index,y_index]=xsinx(arr)


	fig = plt.figure()
	ax = fig.gca(projection='3d')

	# Plot the surface.
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,alpha=0.5,
	                       linewidth=0, antialiased=True)

	# Customize the z axis.
	#ax.set_zlim(-1.01, 1.01)
	ax.view_init(elev=32., azim=46)
	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.set_xlabel(r"$a_1$")
	ax.set_ylabel(r"$a_0$")
	ax.zaxis.set_rotate_label(False)
	ax.set_zlabel(r"$r(a)$",rotation=0)
	plt.show()