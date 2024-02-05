import scipy.io as scio
import scipy.signal as scsig
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
os.getcwd()
# os.chdir('F:/EMG_All/Project_Data_paper/NinaProDB/')
print(os.getcwd())
#  *******************DB1 **********************
'''S1_A1_E1, S1_A1_E2,   S1_A1_E3    S2_A1_E1, S2_A1_E2,  S2_A1_E3   EMG= 101014*10,   Gloves: 101014*22'''

folder_s = './DB1/Data'
folder_t = './DB1_preprocessed_data'
subjects_num = 28
rep_index = 10 #10th column where contains the repetition column value
sti_index = 11 # 11 the column where contains the restimulus colum value
gro_index = 12 # create some groupin dex in 12 th column
if not os.path.exists(folder_t):
    os.makedirs(folder_t)
for s in range(1,subjects_num):
	for e in range(1,4):
		file = '{}/S{}_A1_E{}.mat'.format(folder_s,s,e);
		datamat = scio.loadmat(file)
		_size = np.min([datamat['emg'].shape[0], datamat['rerepetition'].shape[0], datamat['restimulus'].shape[0]]) # calculate the shape of the reading for emg, rerepitition, and restimuls and choose the lowest value
		#DB1: EMG= 222194*10,Repetition: 222194*1,Restimulus: 222194* and  _size=222194
		#print(_size)		a=datamat['emg'][:_size,:] 		print(a.shape)
		'''
		a=datamat['emg'][:_size,:]
		b=datamat['rerepetition'][:_size,:]
		print(a.shape,b.shape)
		d=np.hstack((a,b))
		print(d.shape)
        '''
		data = pd.DataFrame(np.hstack((datamat['emg'][:_size,:],datamat['rerepetition'][:_size,:],datamat['restimulus'][:_size,:]))) #concatenate teh emg, rereptition,restimuls  222194*10, 222194*1, 222194*1
		#print(data.shape)
        #shape of data= 222194*12

        #Create rest repetitions
		data[rep_index].replace(to_replace=0, method='bfill', inplace=True)  #: bfill> backward fille    fill 0 backword cell with forward value  using (1-10)
		rerepetition = data[rep_index].values.reshape((-1,1)) #shape  rerepetition.shape=222194*1 where  data[rep_index].(shape 222194),

		#print(data[rep_index].shape)
		#print(rerepetition.shape)
		#print(rerepetition)
		#print(data.shape)  222194*12

		#Create groups
		data[gro_index] = data[sti_index].replace(to_replace=0, method='bfill')  # fill 0 backword cell with forward value  using (1-10)
		regroup = data[gro_index].values.reshape((-1,1))   #move restimulse value into regroup index 12    shape: 222194*1

		emg = data.loc[:,0:9].values   #0-9 contains the EMG value shape  222194*10
		#print(emg)
		print(data.shape)
		restimulus = data[sti_index].values.reshape((-1,1))	 # shape: 222194*1

		#print(regroup)
		not0 = np.squeeze(np.logical_not(np.isin(regroup, 0)))  # converted all value into True or False . First all are true and last part contains false
		print(not0)
		emg = emg[not0,:]  # Extract only those reading where contains True in not 0 and discard all Fals  reading  now shape # shape (221840, 10) discard 354 reading
		print(emg.shape)
		rerepetition = rerepetition[not0]
		restimulus = restimulus[not0]
		regroup = regroup[not0]
		print(regroup.shape)

		not0 = np.logical_not(np.isin(restimulus, 0)) #221840*1   #contains true and fals, True for all value and  False for 0
		#p=restimulus[not0]
		#p1=restimulus[not0]+12
		#p2=restimulus[not0]+29

		if e==2:     # E1  restimulus unique(restimulu)=12, E2=17,   E3=23
			restimulus[not0] += 12
			regroup += 12
		elif e==3:
			restimulus[not0] += 29
			regroup += 29
		print(restimulus.shape)
		for gesture in np.unique(restimulus):
			g_i = np.isin(restimulus, gesture) # g_i contains  True for all the restimulus index equivalent with gesture other wise false. gesture
            #g_i shape is   (221840, 1), where True for all position where equivalent gesture otherwise False
			file = '{}/subject-{:02d}/gesture-{:02d}/rms'.format(folder_t, int(s), int(gesture))
			if not os.path.exists(file):
				os.makedirs(file)

			if gesture == 0:
				for group in np.unique(regroup):
					_g_i = np.logical_and(np.isin(regroup, group), g_i) #_g_i shape is   (221840, 1), where True for all position where equivalent gesture and equivalent regroup and group otherwise False

					for rep in np.unique(rerepetition):
						r_i = np.isin(rerepetition, rep) #shape r_i is 221840 where True for equivalence of rep with rerepetition otherwise False
						gr_i = np.squeeze(np.logical_and(_g_i, r_i))
						x = emg[gr_i,:]   #extract only for the gesture 0
						y = restimulus[gr_i]
						z = regroup[gr_i]
						w = rerepetition[gr_i]
						scio.savemat(file+'/rep-{:02d}_{:02d}.mat'.format(int(rep), int(z[0])), {'emg':x, 'stimulus':y, 'repetition':w, 'group':z})

			else:
				for rep in np.unique(rerepetition):
					r_i = np.isin(rerepetition, rep)
					gr_i = np.squeeze(np.logical_and(g_i, r_i))
					x = emg[gr_i,:]
					y = restimulus[gr_i]
					z = regroup[gr_i]
					w = rerepetition[gr_i]
					scio.savemat(file+'/rep-{:02d}.mat'.format(int(rep)), {'emg':x, 'stimulus':y, 'repetition':w, 'group':z})
				# break
		# break
	# break
