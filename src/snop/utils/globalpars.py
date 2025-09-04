#
#	Global parameters for FRB HTR polarization analysis
#
#								AB, April 2023

import numpy as np
import pickle as pkl
from collections import namedtuple

#	-------------------		Analysis parameters	----------------------------------------

Raw_time_res_ms		=	1.0e-3/336.0		#	Time resolution of Raw data files in ms
CelebiNchan			=	336					#	Number of channels in CELEBI output files
ZoomLenms			=	300.0				#	Length of the zoomed region in ms	
NatBWmhz			=	-336.0				#	Native bandwidth in MHz			(Positive for USB !)	
MaxWms				=	200.0	#200.0		#	Maximum FRB width in ms for noise estimation ********** SET = 5.0 for 181112		
EdgeFrac			=	0.01				#	Fraction of band-width to reject at either edge		

#	-------------------		Constants	------------------------------------------------

uniG		=	6.67430e-8						#	Universal gravitational constant in CGS
elecE		=	4.8032047e-10					#	Absolute electronic charge in CGS
mE			=	9.1093837e-28					#	Electron mass in CGS
ccC			=	2.99792458e10					#	Speed of light in CGS
pcincm		=	3.0857e18						#	Persec / cm							
wbynu		=	2*np.pi							#	Omega / nu
mSUN		=	1.98847e33						#	Solar mass in grams
radtosec	=	180.0*3600/np.pi				#	Radian in arcsecs
radtopas	=	180.0*3600*1.0e12/np.pi			#	Radian in pico-arcsecs
radsolar	=	6.957e10						#	Solar radius in cm
auincm		=	1.496e13						#	1 AU in cm
intocm		=	2.54							#	1 inch in cm

#	------------------	File Paths	----------------------------------------------------

rawdir		=	'/mnt/ntfsvolume/tempdata/'
htrdir		=	'../frbhtr/'
frbcat		=	'../frbhtr/frbparams.txt'

#	------------------------ Declare FRB data type	------------------------------------

frbhtr		=	namedtuple('frbhtr',['frbname','dm','nchan','ffac','tavgfac','rm','tresms','twinms','subband','refdeg','tmsarr','fmhzarr',\
									 'irms','qrms','urms','vrms',\
									 'it','qt','ut','vt','pt','lt','epts','elts','qfract','ufract','vfract','eqfract','eufract','evfract','lfract','pfract','elfract','epfract',\
									 'chit','psit','echit','epsit','chispec','echispec','psispec','epsispec',\
									 'ispec','qspec','uspec','vspec','eispec','eqspec','euspec','evspec','lspec','pspec','elspec','epspec', \
									 'qfracspec','ufracspec','vfracspec','dqfracspec','dufracspec','dvfracspec','lfracspec','pfracspec','dlfracspec','dpfracspec', \
									 'ids','qds','uds','vds','irmspec','qrmspec','urmspec','vrmspec'])































































