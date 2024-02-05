import torch
import numpy as np
import tensorflow as tf
import random
import os

os.chdir('/home/musa/Musa_Related/PhD/EMG_All/NinaProDB/Musa_NinaPro_Project/')
import preprocessing
from generator import DataGenerator
#from models import getNetwork
from DB2_Mydataset import *
from model.ninapro_network import *
import sys
import json
import scipy
import keras
import tensorflow as tf
from keras import optimizers, initializers, regularizers, constraints
from utils import *
from sklearn import metrics
print(DEFAULT_GENERATOR_PARAMS)
os.getcwd()
os.chdir('/home/musa/Musa_Related/PhD/EMG_All/NinaProDB/')

sys.path.append(os.getcwd())

import torch.optim as optim
import numpy as np
from datetime import datetime
import time
import argparse
import os
#from model.mexcian_network import *

import argparse
#import torch
print(os.getcwd())


parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=2)  # 16
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')  # 1000

parser.add_argument('--patiences', default=500, type=int,
                    help='number of epochs to tolerate no improvement of val_loss')  # 1000

'''
parser.add_argument('--test_subject_id', type=int, default=3,
                    help='id of test subject, for cross-validation')

parser.add_argument('--data_cfg', type=int, default=1,
                    help='0 for 14 class, 1 for 28')

'''
parser.add_argument('--dp_rate', type=float, default=0.1,
                    help='dropout rate')  # 1000

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
folder_process_data = 'Musa_NinaPro_Project/ProcessedData/DB2'
import os
os.environ['PYTHONHASHSEED'] = '0'
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(1234)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
random.seed(12345)
# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see:
#    https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
import tensorflow as tf
from tensorflow import keras

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=0,
    inter_op_parallelism_threads=0
)
session_conf.gpu_options.allow_growth = True
from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
#   https://www.tensorflow.org/api_docs/python/tf/set_random_seed
#tf.set_random_seed(1234)
tf.random.set_seed(1234)
#tf.compat.v1.Session()
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

##############################################################################

import torch.multiprocessing as mp
import torch.distributed as dist 
import torch.nn.functional as F 
import argparse
import warnings

#from mpi4py import MPI
#mpi_rank = MPI.COMM_WORLD.Get_rank()
#mpi_size = MPI.COMM_WORLD.Get_size()

import os
from datetime import timedelta
timeout=timedelta(seconds=86400)
 
os.environ["MASTER_ADDR"]='localhost'
os.environ["MASTER_PORT"]='52277' #netstat -lntu   5567, 21128

dist.init_process_group(backend='nccl',world_size=4, rank=1, store=None, pg_options=None,timeout=timeout)
b=torch.distributed.is_available()
print(b)
print(torch.distributed.is_mpi_available())
print(torch.distributed.is_nccl_available())
print(torch.distributed.is_torchelastic_launched())

'''
def ddp_setup(rank, world_size):
    
    os.environ["MASTER_ADDR"]='localhost'
    os.environ["MASTER_pORT"]='21128' #netstat -lntu   5567, 21128
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

ddp_setup(0, 4)
'''
'''
Parameters
----------
rank :  unique identifier of each process  0 to world_size-1
world_size:: total number of process

    TYPE
    DESCRIPTION.
world_size : TYPE
    DESCRIPTION.

Returns
-------
None.

'''

print('Keras:', keras.__version__)
print('Tensorflow:', tf.__version__)

# 1. Logging
'''
if len(sys.argv) == 4:
    CONFIG_FILE = str(sys.argv[1])
    SUBJECT = int(sys.argv[2])
    TIMESTAMP = int(sys.argv[3])
else:
    print('Expected different number of arguments. {} were given'.format(len(sys.argv) - 1))
    sys.exit()
    '''
CONFIG_FILE='Musa_NinaPro_Project/config_DB2/TCCNet_aot_300.json'
from datetime import datetime
#TIMESTAMP=date
SUBJECT=1
GESTURES=53
currentDateAndTime = datetime.now()

current_date = datetime.now()
TIMESTAMP = current_date .strftime("%Y%m%d%H%M%S")   #20221106210731
print(TIMESTAMP)
#print(TIMESTAMP)

#TIMESTAMP=($(date +"%Y-%m-%d %H:%M:%S %s"))

with open(CONFIG_FILE) as json_file:
    config_params = json.load(json_file)
    print(config_params)     #
outdir='Musa_NinaPro_Project/ProcessedData/DB2/Result'
print(config_params['logging']['log_file'])
LOGGING_FILE_PREFIX = config_params['logging']['log_file'] + '_' + str(TIMESTAMP)
print(LOGGING_FILE_PREFIX )
result_dir='Musa_NinaPro_Project/ProcessedData/DB2/Result/'
if config_params['logging']['enable']:
    LOGGING_FILE = result_dir+'L_' + LOGGING_FILE_PREFIX + '.log'   #CCNet_att_20221106210731
    LOGGING_TENSORBOARD_FILE = result_dir+'L_' + LOGGING_FILE_PREFIX

if config_params['model']['save']:
    MODEL_SAVE_FILE = result_dir+'models/O1_' + LOGGING_FILE_PREFIX + '_{}.json'   # to save the model O1_TCCNet_att_20221106210731_{}.json
    MODEL_WEIGHTS_SAVE_FILE = result_dir+'models/O2_' + LOGGING_FILE_PREFIX + '_{}.h5'  # To save weight file as json file     .h5  O2_TCCNet_att_20221106210731_{}.h5

METRICS_SAVE_FILE = result_dir+'metrics/O3_' + LOGGING_FILE_PREFIX + '_{}.mat'   #save the matrices file  O3_TCCNet_att_20221106210731_{}.mat
print(METRICS_SAVE_FILE)


if not os.path.exists(os.path.dirname(METRICS_SAVE_FILE)):
    try:
        os.makedirs(os.path.dirname(METRICS_SAVE_FILE))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

if not os.path.exists(os.path.dirname(MODEL_SAVE_FILE)):
    try:
        os.makedirs(os.path.dirname(MODEL_SAVE_FILE))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
            
if not os.path.exists(os.path.dirname(LOGGING_TENSORBOARD_FILE)):
    try:
        os.makedirs(os.path.dirname(LOGGING_TENSORBOARD_FILE))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
            


print('Logging file: {}'.format(LOGGING_FILE))   #L_TCCNet_att_20221106210731.log
print('Tensorboard file: {}'.format(LOGGING_TENSORBOARD_FILE)) #L_TCCNet_att_20221106210731
print('Model JSON file: {}'.format(MODEL_SAVE_FILE))  #models/O1_TCCNet_att_20221106210731_{}.json
print('Model H5 file: {}'.format(MODEL_WEIGHTS_SAVE_FILE)) #O2_TCCNet_att_20221106210731_{}.h5
print('Metrics file: {}'.format(METRICS_SAVE_FILE))  # O3_TCCNet_att_20221106210731_{}.mat

print(DEFAULT_GENERATOR_PARAMS)
# 2. Config params generator
PARAMS_TRAIN_GENERATOR = DEFAULT_GENERATOR_PARAMS.copy()
params_gen = config_params['dataset'].get('train_generator', {}).copy()
for key in params_gen.keys():
    PARAMS_TRAIN_GENERATOR[key] = params_gen[key]
print(PARAMS_TRAIN_GENERATOR)
PARAMS_VALID_GENERATOR = DEFAULT_GENERATOR_PARAMS.copy()
params_gen = config_params['dataset'].get('valid_generator', {}).copy()
for key in params_gen.keys():
    PARAMS_VALID_GENERATOR[key] = params_gen[key]

# 3. Initialization
#INPUT_DIRECTORY = '../dataset/Ninapro-DB1-Proc'
INPUT_DIRECTORY = folder_process_data        #link of process dataset
PARAMS_TRAIN_GENERATOR['preprocess_function_1'] = [preprocessing.lpf]
PARAMS_TRAIN_GENERATOR['preprocess_function_1_extra'] = [{'fs': 100}]
PARAMS_TRAIN_GENERATOR['data_type'] = 'rms'
PARAMS_TRAIN_GENERATOR['classes'] = [i for i in range(50)]

PARAMS_VALID_GENERATOR['preprocess_function_1'] = [preprocessing.lpf]
PARAMS_VALID_GENERATOR['preprocess_function_1_extra'] = [{'fs': 100}]
PARAMS_VALID_GENERATOR['data_type'] = 'rms'
PARAMS_VALID_GENERATOR['classes'] = [i for i in range(50)]

print(PARAMS_TRAIN_GENERATOR)

SUBJECTS = config_params['dataset'].get('subjects', [i for i in range(1, 41)])
if np.min(SUBJECTS) <= 0 or np.max(SUBJECTS) >= 41:
    raise AssertionError('Subject IDs should be between 1 and 27 inclusive for DB1. Were given {}\n'.format(SUBJECTS))

PARAMS_TRAIN_GENERATOR.pop('input_directory', '')
PARAMS_VALID_GENERATOR.pop('input_directory', '')

#MODEL = getNetwork(config_params['model']['name'])

mean_train, mean_test, mean_test_3, mean_test_5 = [], [], [], []
mean_cm = []
mean_train_loss, mean_test_loss = [], []

if config_params['logging']['enable']:
    if os.path.isfile(LOGGING_FILE) == False:
        with open(LOGGING_FILE, 'w') as f:
            f.write(
                'TIMESTAMP: {}\n'
                'KERAS: {}\n'
                'TENSORFLOW: {}\n'
                'DATASET: {}\n'
                'TRAIN_GENERATOR: {}\n'
                'VALID_GENERATOR: {}\n'
                'MODEL: {}\n'
                'MODEL_PARAMS: {}\n'
                'TRAIN_PARAMS: {}\n'.format(
                    TIMESTAMP,
                    keras.__version__, tf.__version__,
                    config_params['dataset']['name'], PARAMS_TRAIN_GENERATOR,
                    PARAMS_VALID_GENERATOR,
                    config_params['model']['name'], config_params['model']['extra'],
                    config_params['training']
                )
            )
            f.write(
                'SUBJECT,TRAIN_SHAPE,TEST_SHAPE,TRAIN_LOSS,TRAIN_ACC,TEST_LOSS,TEST_ACC,TEST_TOP_3_ACC,TEST_TOP_5_ACC\n')

print('Subject: {}'.format(SUBJECT))
input_dir = '{}/subject-{:02d}'.format(INPUT_DIRECTORY, SUBJECT)
print(input_dir)

train_generator = DataGenerator(input_directory=input_dir, **PARAMS_TRAIN_GENERATOR)
valid_generator = DataGenerator(input_directory=input_dir, **PARAMS_VALID_GENERATOR)


X_train, Y_train, train_reps,L_=train_generator.get_data()
print(X_train.shape)   # X_train_shape=   2176*14221*12
print(Y_train.shape)   #Y_train_shape= 2176*50
print(Y_train)
print(L_.shape)
print(L_)
import numpy as np

y_train=np.where(Y_train==1)[1]
print(y_train.shape)


X_test, Y_test, test_reps = valid_generator.get_data()     

                             # X test,  Y test

print(X_test.shape)  #shape   X=100*13083, 12  it is for 2 reptation  Y=100*50   
print(Y_test.shape)

y_test=np.where(Y_test==1)[1]

print(y_test.shape)

#reshape of x_train    reading*sample*channel*1
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)

print(X_train.shape[0])
train_dataset = Hand_Dataset(X_train, y_train)   # for 1*8*22*3  label=1
print(train_dataset[1]['skeleton'].shape)
    
print('Call Mydataset for testing :')
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
test_dataset =  Hand_Dataset(X_test, y_test)
print(test_dataset[1]['skeleton'].shape)


print("\nhyperparamter......")
args = parser.parse_args()
print(args)

print("maxican ")

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=False)

val_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=False)

class_num=train_generator.n_classes
print(class_num)


import torch.nn as nn
import torch
import torch.nn.functional as F

#pin_memory=False 
def model_foreward(sample_batched,model,criterion):


    data = sample_batched["skeleton"].float()
    label = sample_batched["label"]
    label = label.type(torch.LongTensor)
    label = label.cuda()
    label = torch.autograd.Variable(label, requires_grad=False)


    score = model(data)

    loss = criterion(score,label)

    acc = get_acc(score, label)

    return score,loss, acc


def get_acc(score, labels):
    score = score.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    outputs = np.argmax(score, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark =False


model = DG_STA(class_num, args.dp_rate, X_train.shape[1])
model = torch.nn.DataParallel(model).cuda()

model_solver = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
   
#........set loss
criterion = torch.nn.CrossEntropyLoss()

train_data_num = 2176
test_data_num = 100
iter_per_epoch = int(train_data_num / args.batch_size)

max_acc = 0
no_improve_epoch = 0
n_iter = 0

#***********training#***********
for epoch in range(args.epochs):
    print("\ntraining.............")
    model.train()
    start_time = time.time()
    train_acc = 0
    train_loss = 0
    for i, sample_batched in enumerate(train_loader):
        n_iter += 1
        #print("training i:",i)
        if i + 1 > iter_per_epoch:
            continue
        score,loss, acc = model_foreward(sample_batched, model, criterion)     #criterionloss function  Model=DG_sta
   
        model.zero_grad()
        loss.backward()
        #clip_grad_norm_(model.parameters(), 0.1)
        model_solver.step()
   
   
        train_acc += acc
        train_loss += loss
   
        #print(i)
   
   
   
    train_acc /= float(i + 1)
    train_loss /= float(i + 1)
   
    print("*** DHS  Epoch: [%2d] time: %4.4f, "
          "cls_loss: %.4f  train_ACC: %.6f ***"
          % (epoch + 1,  time.time() - start_time,
             train_loss.data, train_acc))
    start_time = time.time()
   
    #adjust_learning_rate(model_solver, epoch + 1, args)
    #print(print(model.module.encoder.gcn_network[0].edg_weight))
   
    #***********evaluation***********
    y_pred = []
    y_true = []
    with torch.no_grad():
        val_loss = 0
        acc_sum = 0
        model.eval()
        for i, sample_batched in enumerate(val_loader):
            #print("testing i:", i)
            label = sample_batched["label"]
            score, loss, acc = model_foreward(sample_batched, model, criterion)
            val_loss += loss
   
            if i == 0:
                score_list = score
                label_list = label
            else:
                score_list = torch.cat((score_list, score), 0)
                label_list = torch.cat((label_list, label), 0)
   
   
        val_loss = val_loss / float(i + 1)
        val_cc = get_acc(score_list,label_list)
   
   
        print("*** DHS  Epoch: [%2d], "
              "val_loss: %.6f,"
              "val_ACC: %.6f ***"
              % (epoch + 1, val_loss, val_cc))
   
        #save best model
        if val_cc > max_acc:
            max_acc = val_cc
            no_improve_epoch = 0
            val_cc = round(val_cc, 10)
   
            #torch.save(model.state_dict(),
            #           '{}/epoch_{}_acc_{}.pth'.format(model_fold, epoch + 1, val_cc))
            best_score_list=score_list
            label_list=label_list
            print("performance improve, saved the new model......best acc: {}".format(max_acc))
            
        else:
            no_improve_epoch += 1
            print("no_improve_epoch: {} best acc {}".format(no_improve_epoch,max_acc))
   
        if no_improve_epoch > args.patiences:
            print("stop training....")
            break


'''

timelen=X_train.shape[1]
print(timelen)






# print('Train generator:')
# print(train_generator)
# print('Test generator:')
# print(valid_generator)
print(train_generator.n_classes)
print(train_generator.repetitions)

    # data shape 12734*12      number of channel is 12  
model = MODEL(input_shape=(None, 10),classes=train_generator.n_classes,**config_params['model']['extra'])
#model.summary()

if config_params['training']['optimizer'] == 'adam':
    optimizer = optimizers.Adam(lr=config_params['training']['l_rate'], epsilon=0.001)
elif config_params['training']['optimizer'] == 'sgd':
    optimizer = optimizers.SGD(lr=config_params['training']['l_rate'], momentum=0.9)
    
print(model.summary())

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', top_3_accuracy, top_5_accuracy])



train_callbacks = []
'''
'''

if config_params['logging']['enable']:
    tensorboardCallback = MyTensorboard(log_dir=LOGGING_TENSORBOARD_FILE + "/{}".format(SUBJECT),
                                        batch_size=100,
                                        histogram_freq=10)
    train_callbacks.append(tensorboardCallback)
'''
'''
lrScheduler = MyLRScheduler(**config_params['training']['l_rate_schedule'])
print(lrScheduler)
train_callbacks.append(lrScheduler)

print(train_generator)

history= model.fit(X_train, Y_train,
                    batch_size = 128,
                    epochs = 10, # number of iterations
                    validation_data= (X_test,Y_test), verbose=2)

history = model.fit_generator(train_generator, epochs=config_params['training']['epochs'],   # first calling the generator function(dataAugmentaion) •	Generator function(dataAugmentaion) provides a batch_size of 32 to our .fit_generator() function.•	our .fit_generator() function first accepts a batch of the dataset, then performs 
                              validation_data=(X_test,Y_test), callbacks=train_callbacks, verbose=2) # "train_generator" will call      __augment function from the generator
Y_pred = model.predict(X_test)

y_pred = np.argmax(Y_pred, axis=1)
y_test = np.argmax(Y_test, axis=1)

if config_params['model']['save']:
    # serialize model to JSON
    model_json = model.to_json()
    with open(MODEL_SAVE_FILE.format(SUBJECT), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(MODEL_WEIGHTS_SAVE_FILE.format(SUBJECT))
    print("Saved model to disk")


# Confusion Matrix
# C_{i, j} is equal to the number of observations known to be in group i but predicted to be in group j.
cnf_matrix_frame = metrics.confusion_matrix(y_test, y_pred)
if np.array(mean_cm).shape != cnf_matrix_frame.shape:
    mean_cm = cnf_matrix_frame
else:
    mean_cm += cnf_matrix_frame

mean_train.append(history.history['acc'][-1])
mean_test.append(history.history['val_acc'][-1])
mean_train_loss.append(history.history['loss'][-1])
mean_test_loss.append(history.history['val_loss'][-1])
mean_test_3.append(history.history['val_top_3_accuracy'][-1])
mean_test_5.append(history.history['val_top_5_accuracy'][-1])

if config_params['logging']['enable']:
    with open(LOGGING_FILE, 'a') as f:
        f.write('{},{},{},{},{},{},{},{},{}\n'.format(SUBJECT, train_generator.__len__() * PARAMS_TRAIN_GENERATOR['batch_size'], valid_generator.__len__(),
            mean_train_loss[-1], mean_train[-1], mean_test_loss[-1], mean_test[-1], mean_test_3[-1], mean_test_5[-1]))


metrics_dict = {
    'mean_cm': mean_cm,
    'mean_test': mean_test,
    'mean_test_3': mean_test_3,
    'mean_test_5': mean_test_5,
    'mean_train': mean_train,
    'mean_train_loss': mean_train_loss,
    'mean_test_loss': mean_test_loss
}
scipy.io.savemat(METRICS_SAVE_FILE.format(SUBJECT), metrics_dict)



model_tcn = TCN(input_shape=(None, 10))
'''

