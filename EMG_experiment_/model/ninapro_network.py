import os
import sys


#os.chdir('F:/PhD/EMG_All/Project_Data_paper/NinaProDB/Musa_NinaPro_Project/All_Code/')
os.chdir('/home/musa/Musa_Related/PhD/EMG_All/NinaProDB/Musa_NinaPro_Project/')

print(os.getcwd())
#from model.st_att_layer import *
#from model.st_att_layer import LayerNorm
from model.ninapro_st_att_layer import *

#from model.mexican_st_att_layer.py import LayerNorm
import torch.nn as nn
import torch
import torch.nn.functional as F
ALL=67
BODY=5
FACE=20
HAND=42
BODY_FACE=25
FACE_HAND=62
BODY_HAND=47
#JOINT_NUMBER=ALL
#JOINT_NUMBER=BODY
#JOINT_NUMBER=FACE
#JOINT_NUMBER=HAND
#JOINT_NUMBER=BODY_FACE
#JOINT_NUMBER=FACE_HAND
#JOINT_NUMBER=BODY_HAND
OINT_NUMBER=12

print(JOINT_NUMBER)

class DG_STA(nn.Module):
    def __init__(self, num_classes, dp_rate, tm_len):
        super(DG_STA, self).__init__()
        h_dim = JOINT_NUMBER  #'''67'''***********************************************************
        #h_num= 8  #number of head
        h_num= 8
        
        

        self.input_map = nn.Sequential(
            nn.Linear(1, 128),  #input size, hidden layer size
            nn.ReLU(),
            LayerNorm(128),
            #nn.Linear(128,128),
            #nn.ReLU(),
            
            #nn.Linear(128,128),
            #nn.MaxPool2d((3, 3), stride=(1, 2)),
            #nn.Linear(127,128),
            nn.Dropout(dp_rate),
            nn.Linear(128,128), #newly added
            nn.Dropout(dp_rate)  #newly added
        )
        self.res_map = nn.Sequential(
            nn.Linear(1, 256),  #input size, hidden layer size
            nn.ReLU(),
            LayerNorm(256),
            nn.AvgPool2d((3, 2), stride=(1, 2)),
            nn.Dropout(dp_rate)   #newly added
            
            
        )
        
        #input_size, h_num, h_dim, dp_rate, time_len, domain
        self.s_att = ST_ATT_Layer(input_size=128,output_size= 128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="spatial", time_len = tm_len)

        #self.res_norm=
        self.t_att = ST_ATT_Layer(input_size=128, output_size= 128,h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="temporal", time_len = tm_len)

        self.cls = nn.Linear(128, num_classes)


    def forward(self, x):
        # input shape: [batch_size, time_len, joint_num, 3]

        time_len = x.shape[1]    #8
        joint_num = x.shape[2]   #22

        #reshape x
        original = x.reshape(-1, time_len * joint_num,1)  # 32, 8*22,3   =32*176*3
        
        #first spatial-temporal
        #input map
        inmp = self.input_map(original)  #input=32*176*3
        #inmp=F.pad(inmp, pad=(0, 0, 2, 0), mode='constant', value=0)
        
        #print('inmp shape of inmp',inmp.shape)
        #print('input map of first attention',inmp.shape)     #output 32*176*128
        #spatal
        res=self.res_map(original)
        res=F.pad(res, pad=(0, 0, 2, 0), mode='constant', value=0)
        #print('inmp shape of residual',res.shape)
        
        #print('res :',res.shape,'inmpe :', inmp.shape, 'joint: ', joint_num)
        
        sx = self.s_att(inmp) # #input=32*176*128    output=32*176*256
        #print('x shape of first attention',sx.shape)  
        #temporal
        #sx=res+sx
        #tx = self.t_att(sx) #
        #tx1=res+tx
        #tx1=tx
        
        
        #2nd Stage  temporal-spatial
        #first
        #input map
        #res=self.res_map(original)
        #inmp = self.input_map(original)  #input=32*176*3
        #print('input map of first attention',inmp.shape) 
        #print('input map of second attention',res.shape)     #output 32*176*128
        
        
        #temporal
        #sx = self.t_att(inmp) # #input=32*176*128    output=32*176*256
        #print('x shape of first attention',sx.shape)  
        #temporal
        #sx=self.res_map(original)+sx
        #sx=res+sx  # for this mximum accuracy
        #spatial
        tx = self.s_att(sx) #
        #tx2=res+tx
        tx2=tx
        #print('Second attention',tx.shape) 
        #com=tx1+tx2+res
        com=sx+tx+res
        #print(com.shape)
        x = com.sum(1) / com.shape[1]
        pred = self.cls(x)
        return pred





# MultiBranch spatial temporal-      temporal spatial''' spatial-temporal vs temporal-spatial
 #network for maxican sign language dataset
'''
import os
import sys
os.__package__
#/home/musa/Musa_Related/PhD/Skelton_Depth_VedioSign Language/Multimodal_Oak_D_Camera/Maxican_dataset_DG-STA
#os.chdir('/home/musa/Musa_Related/PhD/Skelton_Depth_VedioSign Language/Multimodal_Oak_D_Camera/Maxican_dataset_DG-STA/')
#os.chdir('/home/musa/Musa_Related/PhD/Skelton_Depth_VedioSign Language/Multimodal_Oak_D_Camera/Maxican_dataset_DG-STA')
os.chdir(r'/home/musa/Musa_Related/PhD/Skelton_Depth_VedioSign Language/Multimodal_Oak_D_Camera/Maxican_dataset_DG-STA')
sys.path.append(os.getcwd())
#os.chdir('/home/musa/Musa_Related/PhD/Skelton_Depth_VedioSign Language/Multimodal_Oak_D_Camera/Maxican_dataset_DG-STA/model')
print(os.getcwd())
#from model.st_att_layer import *
#from model.st_att_layer import LayerNorm
from model.mexican_st_att_layer import *
from model.mexican_st_att_layer import *
#from model.mexican_st_att_layer.py import LayerNorm
import torch.nn as nn
import torch
import torch.nn.functional as F
ALL=67
BODY=5
FACE=20
HAND=42
BODY_FACE=25
FACE_HAND=62
BODY_HAND=47
#JOINT_NUMBER=ALL
#JOINT_NUMBER=BODY
#JOINT_NUMBER=FACE
#JOINT_NUMBER=HAND
#JOINT_NUMBER=BODY_FACE
#JOINT_NUMBER=FACE_HAND
#JOINT_NUMBER=BODY_HAND
OINT_NUMBER=ALL

print(JOINT_NUMBER)

class DG_STA(nn.Module):
    def __init__(self, num_classes, dp_rate):
        super(DG_STA, self).__init__()
        h_dim = JOINT_NUMBER  #'67'***********************************************************
        #h_num= 8  #number of head
        h_num= 8
        
        

        self.input_map = nn.Sequential(
            nn.Linear(3, 128),  #input size, hidden layer size
            nn.ReLU(),
            LayerNorm(128),
            #nn.Linear(128,128),
            #nn.ReLU(),
            
            #nn.Linear(128,128),
            #nn.MaxPool2d((3, 3), stride=(1, 2)),
            #nn.Linear(127,128),
            nn.Dropout(dp_rate),
        )
        self.res_map = nn.Sequential(
            nn.Linear(3, 256),  #input size, hidden layer size
            nn.ReLU(),
            LayerNorm(256),
            nn.AvgPool2d((3, 2), stride=(1, 2))
            
            
        )
        
        #input_size, h_num, h_dim, dp_rate, time_len, domain
        self.s_att = ST_ATT_Layer(input_size=128,output_size= 128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="spatial", time_len = 8)

        #self.res_norm=
        self.t_att = ST_ATT_Layer(input_size=128, output_size= 128,h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="temporal", time_len = 8)

        self.cls = nn.Linear(128, num_classes)


    def forward(self, x):
        # input shape: [batch_size, time_len, joint_num, 3]

        time_len = x.shape[1]    #8
        joint_num = x.shape[2]   #22

        #reshape x
        original = x.reshape(-1, time_len * joint_num,3)  # 32, 8*22,3   =32*176*3
        
        #first spatial-temporal
        #input map
        inmp = self.input_map(original)  #input=32*176*3
        #inmp=F.pad(inmp, pad=(0, 0, 2, 0), mode='constant', value=0)
        
        #print('inmp shape of inmp',inmp.shape)
        #print('input map of first attention',inmp.shape)     #output 32*176*128
        #spatal
        res=self.res_map(original)
        res=F.pad(res, pad=(0, 0, 2, 0), mode='constant', value=0)
        #print('inmp shape of residual',res.shape)
        
        #print('res :',res.shape,'inmpe :', inmp.shape, 'joint: ', joint_num)
        
        sx = self.s_att(inmp) # #input=32*176*128    output=32*176*256
        #print('x shape of first attention',sx.shape)  
        #temporal
        #sx=res+sx
        tx = self.t_att(sx) #
        #tx1=res+tx
        tx1=tx
        
        
        #2nd Stage  temporal-spatial
        #first
        #input map
        #res=self.res_map(original)
        #inmp = self.input_map(original)  #input=32*176*3
        #print('input map of first attention',inmp.shape) 
        #print('input map of second attention',res.shape)     #output 32*176*128
        
        
        #temporal
        sx = self.t_att(inmp) # #input=32*176*128    output=32*176*256
        #print('x shape of first attention',sx.shape)  
        #temporal
        #sx=self.res_map(original)+sx
        #sx=res+sx  # for this mximum accuracy
        #spatial
        tx = self.s_att(sx) #
        #tx2=res+tx
        tx2=tx
        #print('Second attention',tx.shape) 
        com=tx1+tx2+res
        #print(com.shape)
        x = com.sum(1) / com.shape[1]
        pred = self.cls(x)
        return pred


'''

#Temporal and Spatial
'''
import os
os.chdir('/home/musa/Musa_Related/PhD/Skelton_Depth_VedioSign Language/Multimodal_Oak_D_Camera/Maxican_dataset_DG-STA')
print(os.getcwd())
from model.st_att_layer import *
from model.st_att_layer import LayerNorm
import torch.nn as nn
import torch
import torch.nn.functional as F

class DG_STA(nn.Module):
    def __init__(self, num_classes, dp_rate):
        super(DG_STA, self).__init__()
        h_dim = 67
        #h_num= 8  #number of head
        h_num= 8
        
        

        self.input_map = nn.Sequential(
            nn.Linear(3, 128),  #input size, hidden layer size
            nn.ReLU(),
            LayerNorm(128),
            nn.Linear(128,128),
            nn.ReLU(),
            
            #nn.Linear(128,128),
            #nn.MaxPool2d((3, 3), stride=(1, 2)),
            #nn.Linear(127,128),
            nn.Dropout(dp_rate),
        )
        self.res_map = nn.Sequential(
            nn.Linear(3, 256),  #input size, hidden layer size
            nn.ReLU(),
            LayerNorm(256),
            nn.AvgPool2d((3, 2), stride=(1, 2))
            
        )
        
        #input_size, h_num, h_dim, dp_rate, time_len, domain
        self.s_att = ST_ATT_Layer(input_size=128,output_size= 128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="spatial", time_len = 8)

        #self.res_norm=
        self.t_att = ST_ATT_Layer(input_size=128, output_size= 128,h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="temporal", time_len = 8)

        self.cls = nn.Linear(128, num_classes)


    def forward(self, x):
        # input shape: [batch_size, time_len, joint_num, 3]

        time_len = x.shape[1]    #8
        joint_num = x.shape[2]   #22

        #reshape x
        original = x.reshape(-1, time_len * joint_num,3)  # 32, 8*22,3   =32*176*3
        
        #first spatial+temporal
        #input map
        inmp = self.input_map(original)  #input=32*176*3
        #inmp=F.pad(inmp, pad=(0, 0, 2, 0), mode='constant', value=0)
        
        #print('inmp shape of inmp',inmp.shape)
        #print('input map of first attention',inmp.shape)     #output 32*176*128
        #spatal
        res=self.res_map(original)
        res=F.pad(res, pad=(0, 0, 2, 0), mode='constant', value=0)
        #print('inmp shape of residual',res.shape)
        
        
        sx = self.s_att(inmp) # #input=32*176*128    output=32*176*256
        #print('x shape of first attention',sx.shape)  
        #temporal
        sx=res+sx
        tx = self.t_att(sx) #
        tx1=res+tx
       

        com=tx1
        #print(com.shape)
        x = com.sum(1) / com.shape[1]
        pred = self.cls(x)
        return pred

#DG-BiSTA-MRC

'''
'''
import os
os.chdir('/home/musa/Musa_Related/PhD/Skelton_Depth_VedioSign Language/Multimodal_Oak_D_Camera/Maxican_dataset_DG-STA')
print(os.getcwd())
from model.st_att_layer import *
from model.st_att_layer import LayerNorm
import torch.nn as nn
import torch
import torch.nn.functional as F

class DG_STA(nn.Module):
    def __init__(self, num_classes, dp_rate):
        super(DG_STA, self).__init__()
        h_dim = 67
        #h_num= 8  #number of head
        h_num= 8
        
        

        self.input_map = nn.Sequential(
            nn.Linear(3, 128),  #input size, hidden layer size
            nn.ReLU(),
            LayerNorm(128),
            nn.Linear(128,128),
            nn.ReLU(),
            
            #nn.Linear(128,128),
            #nn.MaxPool2d((3, 3), stride=(1, 2)),
            #nn.Linear(127,128),
            nn.Dropout(dp_rate),
        )
        self.res_map = nn.Sequential(
            nn.Linear(3, 256),  #input size, hidden layer size
            nn.ReLU(),
            LayerNorm(256),
            nn.AvgPool2d((3, 2), stride=(1, 2))
            
        )
        
        #input_size, h_num, h_dim, dp_rate, time_len, domain
        self.s_att = ST_ATT_Layer(input_size=128,output_size= 128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="spatial", time_len = 8)

        #self.res_norm=
        self.t_att = ST_ATT_Layer(input_size=128, output_size= 128,h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="temporal", time_len = 8)

        self.cls = nn.Linear(128, num_classes)


    def forward(self, x):
        # input shape: [batch_size, time_len, joint_num, 3]

        time_len = x.shape[1]    #8
        joint_num = x.shape[2]   #22

        #reshape x
        original = x.reshape(-1, time_len * joint_num,3)  # 32, 8*22,3   =32*176*3
        
        #first
        #input map
        inmp = self.input_map(original)  #input=32*176*3
        #inmp=F.pad(inmp, pad=(0, 0, 2, 0), mode='constant', value=0)
        
        #print('inmp shape of inmp',inmp.shape)
        #print('input map of first attention',inmp.shape)     #output 32*176*128
        #spatal
        res=self.res_map(original)
        res=F.pad(res, pad=(0, 0, 2, 0), mode='constant', value=0)
        #print('inmp shape of residual',res.shape)
        
        
        sx = self.s_att(inmp) # #input=32*176*128    output=32*176*256
        #print('x shape of first attention',sx.shape)  
        #temporal
        sx=res+sx
        tx = self.t_att(sx) #
        tx1=res+tx
        
        
        #2nd Stage
        #first
        #input map
        #inmp = self.input_map(original)  #input=32*176*3
        #print('input map of first attention',inmp.shape)     #output 32*176*128
        #spatal
        
        sx = self.s_att(inmp) # #input=32*176*128    output=32*176*256
        #print('x shape of first attention',sx.shape)  
        #temporal
        #sx=self.res_map(original)+sx
        sx=res+sx
        tx = self.t_att(sx) #
        tx2=res+tx
        #print('Second attention',tx.shape) 
        com=tx1+tx2+inmp
        #print(com.shape)
        x = com.sum(1) / com.shape[1]
        pred = self.cls(x)
        return pred
'''
'''
#modified #DG-BiSTA-MRC  # maximum accuracy
import os
os.chdir('/home/musa/Musa_Related/PhD/Skelton_Depth_VedioSign Language/Multimodal_Oak_D_Camera/Maxican_dataset_DG-STA')
print(os.getcwd())
from model.st_att_layer import *
from model.st_att_layer import LayerNorm
import torch.nn as nn
import torch
import torch.nn.functional as F

class DG_STA(nn.Module):
    def __init__(self, num_classes, dp_rate):
        super(DG_STA, self).__init__()
        h_dim = 67
        #h_num= 8  #number of head
        h_num= 8
        
        

        self.input_map = nn.Sequential(
            nn.Linear(3, 128, bias=False),  #input size, hidden layer size
            nn.ReLU(),
            LayerNorm(128),
            #nn.Linear(128,128),
            #nn.ReLU(),
            
            #nn.Linear(128,128),
            #nn.MaxPool2d((3, 3), stride=(1, 2)),
            #nn.Linear(127,128),
            nn.Dropout(dp_rate),
        )
        self.res_map = nn.Sequential(
            nn.Linear(3, 256),  #input size, hidden layer size
            nn.ReLU(),
            LayerNorm(256),
            nn.AvgPool2d((3, 2), stride=(1, 2))
            
        )
        
        #input_size, h_num, h_dim, dp_rate, time_len, domain
        self.s_att = ST_ATT_Layer(input_size=128,output_size= 128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="spatial", time_len = 8)

        #self.res_norm=
        self.t_att = ST_ATT_Layer(input_size=128, output_size= 128,h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="temporal", time_len = 8)

        self.cls = nn.Linear(128, num_classes)


    def forward(self, x):
        # input shape: [batch_size, time_len, joint_num, 3]

        time_len = x.shape[1]    #8
        joint_num = x.shape[2]   #22
        #print('time len:', time_len, 'joint_name:', joint_num)

        #reshape x
        original = x.reshape(-1, time_len * joint_num,3)  # 32, 8*22,3   =32*176*3
        
        #first
        #input map
        #print('shape of original',original.shape)
        
        inmp = self.input_map(original)  #input=32*176*3
        #inmp=F.pad(inmp, pad=(0, 0, 2, 0), mode='constant', value=0)
        
        #print('inmp shape of inmp',inmp.shape)
        #print('input map of first attention',inmp.shape)     #output 32*176*128
        #spatal
        res=self.res_map(original)
        res=F.pad(res, pad=(0, 0, 2, 0), mode='constant', value=0)
        #print('inmp shape of residual',res.shape)
        
        
        sx = self.s_att(inmp) # #input=32*176*128    output=32*176*256
        #print('x shape of first attention',sx.shape)  
        #temporal
        #sx=res+sx
        tx = self.t_att(sx) #
        #tx1=res+tx
        tx1=tx
        
        
        #2nd Stage
        #first
        #input map
        #inmp = self.input_map(original)  #input=32*176*3
        #print('input map of first attention',inmp.shape)     #output 32*176*128
        #spatal
        
        sx = self.s_att(inmp) # #input=32*176*128    output=32*176*256
        #print('x shape of first attention',sx.shape)  
        #temporal
        #sx=self.res_map(original)+sx
        #sx=res+sx
        tx = self.t_att(sx) #
        #tx2=res+tx
        tx2=tx
        #print('Second attention',tx.shape) 
        
        #3rd Stage
        sx = self.s_att(inmp) # #input=32*176*128    output=32*176*256
        #print('x shape of first attention',sx.shape)  
        #temporal
        #sx=self.res_map(original)+sx
        #sx=res+sx
        tx = self.t_att(sx) #
        #tx2=res+tx
        tx3=tx
        
        
        com=tx1+tx2+tx3+inmp
        #print(com.shape)
        x = com.sum(1) / com.shape[1]
        pred = self.cls(x)
        return pred
'''
