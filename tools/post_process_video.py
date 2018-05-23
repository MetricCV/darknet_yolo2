import json 
import pandas as pd
import numpy as np
# This functions takes the output of process_video and count the number of appearances of every category, 
# in this counting process we consider a tolerance between two different frames before assuming that the category 
# has dissapear from the image.
# Inputs:
# ======
# dataname (str) path to results from process_video
# maxtolfps (int) number of frames that will accept without counting another appearance.
# fpsrate (int) number of frames per second of the video
# frameofbeginningsecondhalf (int) integer indicating where does the second half begin (in the video) 
	# it can be calculated from [frame=fpsrate*(minute_of_finish*60+secon_of_finish)].
# 
# Outpus:
# =======
# Data_from_Video (datafile.csv) to every category we calculate
	# Numero Total de Apariciones, Numero de Apariciones durante Primer Tiempo,
	# Numero de Apariciones durante Segundo Tiempo,	Tiempo Maximo en Pantalla (s),
	# Tiempo Minimo en Pantalla (s), Tiempo Promedio en Pantalla (s), Tiempo Total en Pantalla (s)
dataname="/mnt/backup/VA/futbol_mexico/Monterrey_vs_Tigres_C2017_small_output_yolo.json"
maxtolfp=13
frameofbeginningsecondhalf=71750#(25*(47*60+8)) 
fpsrate=25
data=json.load(open(dataname,"r"))
categories=list(data.keys())
timeintv={}
maxintervalintv={}
minintervalintv={}
numofapperances={}
numofapperances_firsthalf={}
numofapperances_secondhalf={}
avtimeinttv={}
for i in categories:
	if len(data[i])>1:
		timeintv[i]=0
		maxintervalintv[i]=0
		minintervalintv[i]=180000
		numofapperances[i]=0
		numofapperances_firsthalf[i]=0
		numofapperances_secondhalf[i]=0
		prevfr=data[i][0]
		bfrsecuence=data[i][0]#frame id que identifica el comienzo de la subsecuencia de frames
		for j in range(1,len(data[i])):
			deltafr=data[i][j]-prevfr
			if deltafr>=maxtolfp:#si la diferencia es de mas 13 fps
				numofapperances[i]+=1
				maxintervalintv[i]=max(maxintervalintv[i],prevfr-bfrsecuence+1)
				minintervalintv[i]=min(minintervalintv[i],prevfr-bfrsecuence+1)
				timeintv[i]+=(prevfr-bfrsecuence)+1#
				prevfr=data[i][j]
				bfrsecuence=data[i][j]
				if prevfr>=frameofbeginningsecondhalf:
					numofapperances_secondhalf[i]+=1
				else:
					numofapperances_firsthalf[i]+=1
			else:	
				prevfr=data[i][j]
		avtimeinttv[i]=np.around(timeintv[i]/numofapperances[i],decimals=2)
		timeintv[i]=np.around(timeintv[i]/fpsrate,decimals=2)
		maxintervalintv[i]=np.around(maxintervalintv[i]/fpsrate,decimals=2)
		minintervalintv[i]=np.around(minintervalintv[i]/fpsrate,decimals=2)
		avtimeinttv[i]=np.around(avtimeinttv[i]/fpsrate,decimals=2)	
		print("cambio de categoria", i)
		print("tiempo categoria", timeintv[i])
		print("maximo tiempo categoria",maxintervalintv[i])
		print("minimo tiempo categoria",minintervalintv[i])	
		print("timepo promedio de intervencion", avtimeinttv[i])
		print("Numero de apariciones en el primer tiempo", numofapperances_firsthalf[i])
		print("Numero de apariciones en el segundo tiempo", numofapperances_secondhalf[i])
		print("Numero total de apariciones", numofapperances[i])
dictionary={"Tiempo Total en Pantalla (s)": timeintv,
"Tiempo Maximo en Pantalla (s)": maxintervalintv,
"Tiempo Minimo en Pantalla (s)": minintervalintv,
"Tiempo Promedio en Pantalla (s)": avtimeinttv,
"Numero Total de Apariciones ": numofapperances,
"Numero de Apariciones durante Primer Tiempo": numofapperances_firsthalf,
"Numero de Apariciones durante Segundo Tiempo": numofapperances_secondhalf}	
df=pd.DataFrame(dictionary, index=categories)		
df.to_csv('../results_arpon/Data_from_Video.csv',sep=",",encoding="utf-8")