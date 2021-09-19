#import matplotlib.pyplot as plt

def draw_scatter_3d(plt,x,y,z,value,x_label,y_label,z_label):
	fig = plt.figure()
	ax = plt.axes(projection ='3d')
	imag = ax.scatter(x,y,z,c=value,cmap=plt.hot())
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	ax.set_zlabel(z_label)
	fig.colorbar(imag)
	return plt,fig,ax
