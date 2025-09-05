import numpy as np
import os
from ..utils.globalpars import *


def generate_dynspec(t_ser, n):
	#	Creates a dynamic spectrum at the highest time resolution from the given voltage time series.
    #
    #	Inputs		input time series of voltages
    #				number of channels
    #
    #	Returns		dynamic spectrum of voltages
    
    dynspec = np.zeros((int(t_ser.shape[0] / n), n), dtype=np.complex64)
    for i in range(int(t_ser.shape[0] / n)):
        dynspec[i, :] = np.fft.fft(t_ser[i * n : (i + 1) * n])

    return(dynspec)

def voltage_ds(frbname, X, Y, dm, nchan, outdir):

	#	Construct voltage dynamic spectra for X and Y

	dslen		=	0
	
	if(os.path.exists(X) and os.path.exists(Y)):
		xvdat		=	np.load(X,mmap_mode='r')
		yvdat		=	np.load(Y,mmap_mode='r')
		xvdspec		=	generate_dynspec(xvdat, nchan)
		yvdspec		=	generate_dynspec(yvdat, nchan)
		#np.save("{}reduced/{}_x_vds_{}_{}.npy".format(outdir,frbname,dm,nchan),xvdspec.T)
		#np.save("{}reduced/{}_y_vds_{}_{}.npy".format(outdir,frbname,dm,nchan),yvdspec.T)
		dslen		=	xvdspec.shape[0]
		del(xvdat)
		del(yvdat)
	else:
		print("voltage_ds ---- PANIC - File(s) not found!")
		print("\n******* Have you polcaled ? ******\n")

	return xvdspec.T, yvdspec.T



def calculate_stokes_unnormalised(x, y):
    # 	lambda functions for each of the Stokes parameters
    #
    #	Inputs		X & Y voltage time series
    #
    #	Returns		Stokes time series (I, Q, U, V)
        
    stokes = {
        "i": lambda x, y: np.abs(x) ** 2 + np.abs(y) ** 2,
        "q": lambda x, y: np.abs(y) ** 2 - np.abs(x) ** 2,
        "u": lambda x, y: 2 * np.real(np.conj(x) * y),
        "v": lambda x, y: 2 * np.imag(np.conj(x) * y),
    }
    stks = []
    
    #print("*******************************")
    #print("\n Negating Q to conform to the world of pulsars !!! \n")
    #print("*******************************")
	
    for idx, stk in enumerate(["i", "q", "u", "v"]):
        par = stokes[stk](x, y)
        par_norm = par
        del par
        par = par_norm.transpose()
        del par_norm
        stks += [par]
    
    return(stks)


def gen_stokes_ds(frbname, x, y, dm, nchan, outdir):

	#	Construct dynamic spectra for I, Q, U, V

#	if(os.path.exists(x) and os.path.exists(y)):
	#xvds	=	np.load(x)
	#yvds	=	np.load(y)
	#pcals	=	np.loadtxt(polcalfile)
	stksds	=	np.zeros((4,x.shape[0],y.shape[1]), dtype=float)
	
	for cc in range(0,nchan):
		stkst	=	calculate_stokes_unnormalised(x[cc], y[cc])
		for i in range(0,4):
			stksds[i,cc]	=	stkst[i]	
	
	np.save("{}{}_stks_ds_{}_{}_avg_1_1.npy".format(outdir,frbname,dm,nchan),stksds)
	print("Saved: {}{}_stks_ds_{}_{}_avg_1_1.npy".format(outdir,frbname,dm,nchan))
	#print(stksds.shape)
	del(stksds)
	del(x)
	del(y)
#	else:
#		print("gen_stokes_ds ---- PANIC - File(s) not found!")

	return #(nchan*Raw_time_res_ms)