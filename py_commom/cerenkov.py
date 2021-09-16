
import sympy as sp
import numpy as np
def caculate_length_inns(distance,theta,ck_angle,track_length,timewindow=3):
	area = check_which_area(distance,theta,ck_angle,track_length)
	c = 3.0*10**8; Rindex = 1.49; 
	print("area == ",area)
	x=sp.Symbol("x")
	if area == "backward":
		solution=sp.solve(x/c+(distance**2+x**2-2*distance*x*np.cos(np.pi/180.0*theta))**0.5/(c/Rindex)-distance/(c/Rindex)-timewindow*10**(-9))
		print("solution1 == ",solution)


	if area == "forward":
		t_f = track_length/c+(distance**2+track_length**2-2*distance*track_length*np.cos(np.pi/180.0*theta))**0.5/(c/Rindex)
		print("t_f = ",t_f,"distance = ",distance,"theta = ",theta,"Rindex = ",Rindex,"timewindow = ",timewindow,"c = ",c,"track_length = ",track_length)
		#L=c*(t_f+timewindow*10**(-9))
		#print("L == ",L)	
		#solution=sp.solve(x/c+(distance**2+x**2-2*distance*x*np.cos(np.pi/180.0*theta))**0.5/(c/Rindex)-t_f-timewindow*10**(-9))
		solution=sp.solve(x/c+(distance**2+x**2-2*distance*x*np.cos(np.pi/180.0*theta))**0.5/(c/Rindex)-t_f-timewindow*10**(-9))
		print("solution2 ==",solution)


	if area == "mid":
		t_f_1=(distance*np.cos(np.pi/180.0*theta)-distance*np.sin(np.pi/180.0*theta)/np.tan(np.pi/180*ck_angle))/c
		t_f_2=(distance*np.sin(np.pi/180.0*theta)/np.sin(np.pi/180.0*ck_angle))/(c/Rindex)
		t_f = t_f_1 + t_f_2
		print("t_f ==",t_f)
		solution=sp.solve(x/c+(distance**2+x**2-2*distance*x*np.cos(np.pi/180.0*theta))**0.5/(c/Rindex)-t_f - timewindow*10**(-9))
		print("solution3 ==",solution)

	if solution == []:
		print("warning!! distance = {},theta = {},area = {}".format(distance,theta,area))
		return 0

	if (solution[0] < 0 )and (solution[1] > track_length):
		return track_length
	if (solution[0] > 0 )and (solution[1] < track_length):
		return solution[1]-solution[0]
	if (solution[1] < track_length) and (solution[1] > 0) and (solution[0] < 0):
		return solution[1]
	if (solution[0] > 0 ) and (solution[0]<track_length) and (solution[1] > track_length):
		return track_length-solution[0]	
	return 0

def check_which_area(distance,theta,ck_angle,track_length):
	'''
	                   /
                          /
                         /
                        / 
	               /	    
                      /
                     /\ck_angle
                    /  \
                   /-----------------------------------------
		start                  track_length         end
	'''
	if theta >= ck_angle:
		return "backward"
	if theta < ck_angle:
		x1 = distance*np.sin(np.pi/180.0*(ck_angle-theta))/np.sin(np.pi/180.0*(180.0-ck_angle))
	if x1 <= track_length:
		return "mid"
	if x1 > track_length:
		return "forward"	

def draw_d_theta(ck_angle=47.84,track_length=10,timewindow=3):
	d=np.arange(0.5,35,0.5)
	theta=np.arange(0,180,5)
	D,Theta=np.meshgrid(d,theta)
	W=np.zeros((D.shape[0],D.shape[1]),dtype="float32")
	for i in range(D.shape[0]):
		for j in range(D.shape[1]):
			print("D[{},{}]=={}".format(i,j,D[i,j]))
			print("Theta[{},{}]=={}".format(i,j,Theta[i,j]))
			W[i,j] = caculate_length_inns(D[i,j],Theta[i,j],ck_angle,track_length,timewindow)
			print("length=",W[i,j])
 	
	import matplotlib.pyplot as plt
	fig,ax=plt.subplots()
	p=ax.pcolor(Theta,D,W)

	d_constant=np.arange(0,35,0.1)
	theta_con=np.ones_like(d_constant)*ck_angle
	ax.plot(theta_con,d_constant,color="red",label="mid/backward")

	bound_theta_1 = np.arange(0,ck_angle,2)
	bound_d_1 = track_length*np.tan(np.pi/180.0*ck_angle)/(np.tan(np.pi/180.0*ck_angle)*np.cos(np.pi/180.0*bound_theta_1)-np.sin(np.pi/180.0*bound_theta_1))
	ax.plot(bound_theta_1,bound_d_1,color="green",label="forward/mid")	

	cb = fig.colorbar(p,ax=ax)
	cb.set_label("length_in_{}ns(m)".format(timewindow))
	ax.set_title("tracklength={}m,ck_angle={}deg,timewindow={}ns".format(track_length,ck_angle,timewindow))
	ax.set_ylabel("d(m)")
	ax.set_xlabel(r"$\theta$(deg)")
	ax.set_ylim(0,35)
	ax.legend()
	plt.show()
	
def draw_simple_time(d,theta,Rindex):
	import scipy.constants as C
	ck_angle=np.arccos(1.0/Rindex)/np.pi*180.0
	x=np.arange(-200,200,1)
	t=x/C.c + (d**2+x**2-2*d*x*np.cos(np.pi/180.0*theta))**0.5/(C.c/Rindex)
	import matplotlib.pyplot as plt
	fig,ax=plt.subplots()
	ax.plot(x,t,color="red")
	plt.show()

if __name__ == "__main__":
	print("hello,world")
	draw_simple_time(d=23.5,theta=28,Rindex=1.49)
	a=caculate_length_inns(distance=23.5,theta=28,ck_angle=47.84,track_length=3,timewindow=3)
	#print("length==",a)
	draw_d_theta(track_length=2.5)

