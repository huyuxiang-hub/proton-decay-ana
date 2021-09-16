import numpy as np
from root_operations import *
import ROOT
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True,threshold=np.inf)
import argparse
import sys

def get_args():
	'''
	return some argument.
	'''
	parser = argparse.ArgumentParser(description='count to script')
	parser.add_argument("--input",default="SimEvent.root")
	parser.add_argument("--evtid",default=0,type=int)
	parser.add_argument("--pmtmapfile",default="lpmt.txt")
	parser.add_argument("--mode",default=0,type = int ) # 0 for cerenkov,1 for first hit in ns
	parser.add_argument("--debug",default=0,type = int )
	args = parser.parse_args()
	
	return args

def init_pmt_info(pmtid , hittime , pmtmap):
	'''
	'''
	pmt_number=pmtmap.shape[0]
	a = np.zeros((pmt_number,15),dtype='float64')
	for i in range(pmt_number):
		b=np.sort(hittime[pmtid == i])
		if b.size > 10:
			a[i,:10]=b[:10]
		else:
			a[i,:b.size]=b
		a[i,10:15] = pmtmap[i,1:] 
		
       
	#print(a[100:1000,:])
	
	return a
def draw_eachpmt_hittime(pmtid , hittime , pmtmap,isCerenkov,isOriginalOP,debug=0):
	pmt_number=pmtmap.shape[0]
	if debug:
		pmtid=np.array([1,1,1,1,1,1,1,2,2,2,2,2])
		hittime=np.array([3,4,5,6,7,8,9,1,2,3,4,5])
		isCerenkov=np.array([0,1,0,1,0,1,0,1,0,1,0,1])
		isOriginalOP=np.array([1,1,1,1,1,1,1,1,1,1,1,1])
	
	id_to_hittime={}
	#a = np.zeros((pmt_number,20),dtype='float32')
	for i in range(pmt_number):
		b=np.sort(hittime[pmtid == i])
		id_to_hittime[i]=b
	
	pmtid_cerenkov=[]
	for i in range(isCerenkov.size):
                #if (isCerenkov[i] == 1) and (pmtid[i]<17612):
		if (isCerenkov[i] == 1) and (isOriginalOP[i] == 1) and (pmtid[i]<17612):
			pmtid_cerenkov.append(pmtid[i])
	#print("pmtid_cerenkov==",pmtid_cerenkov)
	pmtid_cerenkov = np.array(list(pmtid_cerenkov))
	from collections import Counter
	id_number=Counter(pmtid_cerenkov)
	for k in id_number.keys():
		if id_number[k] == 5:
			print(5.0/len(id_to_hittime[k]))
			print(id_to_hittime[k])		

def draw_fraction_of_first_time(pmtid , hittime , pmtmap,isCerenkov, isOriginalOP,timewindow=3,debug=0):
	id_to_hitnum={}
	if debug:
		pmtid=[1,1,1,1,1,2,2,2,3,3,3]
		hittime=[100,101,107,101,101,91,90,100,56,57,60]
		print("pmtid.size == ",len(pmtid))
	for i in range(len(pmtid)):
		id_to_hitnum[pmtid[i]]=[]
	for i in range(len(pmtid)):
		if (isCerenkov[i] > -1 ) and (isOriginalOP[i] > -1):
			id_to_hitnum[pmtid[i]].append(hittime[i])
	x=[];y=[];z=[];f=[]
	for k in id_to_hitnum.keys():
		if k > 17613:
			continue
		if id_to_hitnum[k] == []:
			continue
		id_to_hitnum[k].sort()
		tmp=np.array(id_to_hitnum[k])
		f.append((np.sum(tmp<(tmp[0]+timewindow)))/(tmp.size*1.0))
		z.append(pmtmap[k,3])
		y.append(pmtmap[k,2])
		x.append(pmtmap[k,1])
	if debug:
		print("fraction==",f)
	fig = plt.figure()
	ax = plt.axes(projection ='3d')
	imag = ax.scatter(x,y,z,c=f,cmap=plt.hot())
	ax.set_xlabel("x(mm)")
	ax.set_ylabel("y(mm)")
	ax.set_zlabel("z(mm)")
	fig.colorbar(imag)
	plt.show()
	
		       

def draw_first_hit(pmtid , hittime , pmtmap,isCerenkov, isOriginalOP,debug=0):
	id_to_hitnum={}
	
	if debug:
		pmtid=[1,1,2,2,3,3]
		hittime=[100,101,91,90,56,57]
		print("pmtid.size == ",len(pmtid))

	for i in range(len(pmtid)):
		id_to_hitnum[pmtid[i]]=[]
	for i in range(len(pmtid)):
		#print("pmtid[i]",pmtid[i])
		if (isCerenkov[i] > -1 ) and (isOriginalOP[i] > -1):
			id_to_hitnum[pmtid[i]].append(hittime[i])
	#print(id_to_hitnum)
	
	x=[];y=[];z=[];w=[]
	for k in id_to_hitnum.keys():
		
		if k > 17613:
			continue
		id_to_hitnum[k].sort()
		print("k=",k)
		print("id_to_hitnum",id_to_hitnum[k])
		z.append(pmtmap[k,3])
		y.append(pmtmap[k,2])
		x.append(pmtmap[k,1])
		if id_to_hitnum[k] == []:
			w.append(0)
		else:
			w.append(id_to_hitnum[k][0])
	#print(id_to_hitnum)
	fig = plt.figure()
	ax = plt.axes(projection ='3d')
	imag = ax.scatter(x,y,z,c=w,cmap=plt.hot())
	ax.set_xlabel("x(mm)")
	ax.set_ylabel("y(mm)")
	ax.set_zlabel("z(mm)")
	fig.colorbar(imag)
	plt.show()

	'''
	bins=np.arange(80, 200, 1)
	ah = np.histogram(pmt_info[:,0], bins)
	fig, ax = plt.subplots()
	ax.semilogy( bins[:-1], ah[0], label="first-hit-time",drawstyle="steps-post" )
	ax.set_xlabel("time(ns)",loc="right")
	ax.set_ylabel("Events",loc="top")  #loc="bottom","top"
	ax.set_title("mu-sample/analysis.py draw_first_hit")
	ax.legend()
	plt.show()
	'''
	#fig.savefig("firsthittime")

def draw_cerenkov(pmtid,isCerenkov,isOriginalOP,pmtmap):
	'''
	pmtid: numpy array. it is from the branch pmtID of evt tree.
	isCerenkov: numpy array. it is from the branch isCerenkov of evt tree.
	pmtmap: numpy array. it is from lpmt.txt
	function: draw the pmt distribution which is hitted by cerenkov light 
	'''
	pmtid_cerenkov=[]
	for i in range(isCerenkov.size):
		#if (isCerenkov[i] == 1) and (pmtid[i]<17612):
		if (isCerenkov[i] == 1) and (isOriginalOP[i] == 1) and (pmtid[i]<17612):
			pmtid_cerenkov.append(pmtid[i])
	pmtid_cerenkov = np.array(list(pmtid_cerenkov))
	theta=[]
	phi=[]
	for i in range(pmtid_cerenkov.size):
		theta.append(pmtmap[pmtid_cerenkov[i],5])
		phi.append(pmtmap[pmtid_cerenkov[i],4])
	theta=np.array(list(theta))
	phi=np.array(list(phi))
	if(1):
		draw_cerenkov_in3d(pmtid_cerenkov,pmtmap)
		#return
	
	phi_bins = np.arange(-180, 180, 4)
	theta_bins = np.arange(-90,90, 2)
	plt.hist2d(phi, theta, bins =[phi_bins, theta_bins])
	plt.colorbar()
	plt.xlabel("$\phi$(deg)")
	plt.ylabel(r"$\theta$(deg)")
	plt.title("isCerenkov[i] == 1 and isOriginalOP[i] == 1 and pmtid[i]<17612")
	plt.show()
	print("phi.shape ==  ", phi.shape)
	print("pmtid_cerenkov == ",pmtid_cerenkov.shape)
def draw_cerenkov_in3d(pmtid_cerenkov,pmtmap):
	from collections import Counter
	id_number=Counter(pmtid_cerenkov)
	x=[]; y=[]; z=[]; w=[];
	for k in id_number.keys():
		z.append(pmtmap[k,3])
		y.append(pmtmap[k,2])
		x.append(pmtmap[k,1])
		w.append(id_number[k])
	fig = plt.figure()
	ax = plt.axes(projection ='3d')
	imag = ax.scatter(x,y,z,c=w,cmap=plt.hot())
	ax.set_xlabel("x(mm)")
	ax.set_ylabel("y(mm)")
	ax.set_zlabel("z(mm)")
	fig.colorbar(imag)
	plt.show()
	
def modify_pmtmap(pmtmap):
	'''
	argument: pmtmap:this is an numpy array which save the lpmt.txt info
	return: this function will modify the pmtmap' theta and phi, just like a world map.
	'''
	for i in range(pmtmap.shape[0]):
		if pmtmap[i,5] > 180:
			pmtmap[i,5]=pmtmap[i,5]-360
		pmtmap[i,4]=90-pmtmap[i,4]
		#print("i==",i)
		#print("phi==",pmtmap[i,4])
		#print("theta==",pmtmap[i,5])
def draw_pmt_hit_inns(pmtid, hittime , pmtmap , timewindow = 6,debug=0):
	if debug:
		pmtid=np.array([1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6])
		hittime=np.array([101,102,103,104,105,106,101.2,102,103,104,105,106,107,102,103,104,105,106])
	a = init_pmt_info(pmtid , hittime , pmtmap)
	if debug:
		print("timewindow = ",timewindow)
		print("hittime = ",hittime)
		print("pmtid = ",pmtid)
		print("pmtmap= ",pmtmap[:7])
	first_hittime=a[:,0]
	if debug:
		print("first_hittime==",first_hittime[:7])
	pmt_hitnumber_inns=np.zeros((pmtmap.shape[0],1),dtype="int32")
	for i in range(hittime.size):
		if pmtid[i] >= 17612:
			continue
		if (hittime[i]-first_hittime[pmtid[i]]) < timewindow :
			#print("hittime[",i,"]=",hittime[i])
			#print("first_hittime[pmtid[i]]==",first_hittime[pmtid[i]])
			pmt_hitnumber_inns[pmtid[i]]+=1
	theta=[]
	phi=[]
	if debug:
		print("pmt_hitnumber_inns == ",pmt_hitnumber_inns)
	for i in range(pmt_hitnumber_inns.shape[0]):
		for k in range(pmt_hitnumber_inns[i,0]):
			theta.append(pmtmap[i,4])
			phi.append(pmtmap[i,5])
	theta=np.array(list(theta))
	phi=np.array(list(phi))

	phi_bins = np.arange(-180, 180, 4)
	theta_bins = np.arange(-90,90, 2)
	plt.hist2d(phi, theta, bins =[phi_bins, theta_bins])
	plt.colorbar()
	plt.xlabel("$\phi$(deg)")
	plt.ylabel(r"$\theta$(deg)")
	plt.title("draw_pmt_hit_inns")
	plt.show()
	
	
def caculate_pe_of_prd(m_p=938.272,m_e=0.510998,m_pi0=134.9770 ):
	'''
	this is about caculation the final momenta of e+
	'''
	print("physics constant website:  https://pdg.lbl.gov/2020/reviews/rpp2020-rev-phys-constants.pdf")
	print("p --> e+ + pi0 : m_p=%f,m_e=%f,m_pi0=%f" %(m_p,m_e,m_pi0))
	print("pe^2 = ((m_p^2+m_e^2-m_pi0^2)/(2*m_p))^2-m_e^2")
	p_e=(((m_p**2+m_e**2-m_pi0**2)/(2*m_p))**2-m_e**2)**(0.5)
	print("p_e ==",p_e)


if __name__ == "__main__":
	print("hello world !! welcome to proton decay analysis")
	args=get_args()
	if args.mode == 3:
		caculate_pe_of_prd()
		sys.exit()
	a = np.loadtxt(args.pmtmapfile)
	print("a.shape == ",a.shape)
	modify_pmtmap(a)
		
	myfile, mytree, entries = readTree(args.input,tree='evt')
	entry=args.evtid
	hitTime = getBranchData(tree = mytree, entry = entry, branch_name='hitTime')
	hittime = np.array(list(hitTime))
	isCerenkov = np.array(list(getBranchData(tree = mytree, entry = entry , branch_name='isCerenkov')))     
	#print(isCerenkov)
	pmtid=np.array(list(getBranchData(tree = mytree, entry = entry , branch_name='pmtID')))
	isOriginalOP=np.array(list(getBranchData(tree = mytree, entry = entry , branch_name='isOriginalOP')))
	
	if args.mode == 1:
		draw_pmt_hit_inns(pmtid = pmtid, hittime = hittime  , pmtmap = a , timewindow = 2,debug=args.debug)
	if args.mode == 0:
		draw_cerenkov(pmtid,isCerenkov,isOriginalOP,a)
 
	#print(hittime.shape)
	#print(pmtid.shape)
	
	if args.mode == 2:
		pmt_info = init_pmt_info(pmtid=pmtid,hittime=hittime,pmtmap=a)
		draw_first_hit(pmtid=pmtid,hittime=hittime,pmtmap=a,debug=args.debug,isCerenkov=isCerenkov,isOriginalOP=isOriginalOP)
	if args.mode == 4:
		draw_eachpmt_hittime(pmtid=pmtid , hittime=hittime , pmtmap=a ,isCerenkov=isCerenkov,isOriginalOP=isOriginalOP,debug=args.debug)
	if args.mode == 5:
		draw_fraction_of_first_time(pmtid=pmtid , hittime=hittime , pmtmap=a,isCerenkov=isCerenkov, isOriginalOP=isOriginalOP,timewindow=3,debug=args.debug)


