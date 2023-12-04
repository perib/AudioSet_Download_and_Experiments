#RESTART AND CLEAR BECAUSE THE TENSORFLOW GRAPH PERSISTS
'''
audiosetDL

Pedro Ribeiro
Josh McDermotts Lab
MSRP BIO 2017

documentation: https://github.mit.edu/ribeirop/AudioSetDL/blob/master/README.md

'''

import os
import numpy as np
from csv import reader
import csv
from scipy.io import wavfile
import scipy
import h5py
import glob
import scipy.signal as signal
import argparse
import h5py
import sys
import tensorflow as tf
import time
import resource

import matplotlib.pyplot as plt

from IPython.core.display import HTML, display
from scipy.io import wavfile

from tensorflow.python.client import timeline

sys.path.insert(0, './HELPER_PROGRAMS/tfcochleagram')
import tfcochleagram
#for debugging and sanity checks
def wavPlayer(filepath):
    """ will display html 5 player for compatible browser

    Parameters :
    ------------
    filepath : relative filepath with respect to the notebook directory ( where the .ipynb are not cwd)
               of the file to play

    The browser need to know how to play wav through html5.

    there is no autoplay to prevent file playing when the browser opens
    """

    src = """
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>Simple Test</title>
    </head>

    <body>
    <audio controls="controls" style="width:600px" >
      <source src="files/%s" type="audio/wav" />
      Your browser does not support the audio element.
    </audio>
    </body>
    """%(filepath)
    display(HTML(src))
    
#creates a dictionary for easy access from labels to indeces
def genLabelDictionary(path):
    f = open(path)
    dic = {}
    f.readline()    
    for line in reader(f):
        base = line[1]
        dic[base] = line[0]
        dic[line[0]] = line[2]
    
    return dic  

def MEAN(outputfile = "/om/user/ribeirop/audiosetDL/balanced_stripped/",wavHDF5file = "/om/user/ribeirop/audiosetDL/balanced_stripped/balanced_train_segments.hdf5"):
    #create the files for the databases  
    basename = os.path.splitext(os.path.basename(wavHDF5file))[0]
    
    Storage = h5py.File("{0}{1}_Coch.hdf5".format(outputfile,basename),"r")

    cochSet = Storage["/coch"]
        
    cochlabelSet = Storage["/labels"]
    
    totalNumber = Storage.attrs.get("size")
    
    
    sums = np.zeros(cochSet[0].shape, dtype=np.float64)
    batchSize = 10000
    current_start = 0
    while(current_start + batchSize <= totalNumber):
        end = current_start + batchSize
        av = np.mean(cochSet[current_start:end],axis = 0, dtype=np.float64)  
        sums = np.add(sums , np.multiply(av , np.divide((end-current_start),(totalNumber)),dtype=np.float64))
        current_start = current_start + batchSize
    
    if(current_start < totalNumber):
        end = current_start + (totalNumber-current_start)
        av = np.mean(cochSet[current_start:end],axis = 0, dtype=np.float64)  
        sums = np.add(sums , np.multiply(av , np.divide((end-current_start),(totalNumber)),dtype=np.float64))
        current_start = current_start + batchSize
    
    
    
    print("final", sums)
    np.save("{0}{1}_average.npy".format(outputfile,basename),sums)
    
    Storage.close()
    print("done averaging")
    

# -o', '--outputPath' : path to where you want the final HDF5 file to be stored. 
# '-w', '--wavHDF5file' : path to the HDF5 file containing the wavs created by Wav_into_HDF5
def main(outputfile = "/om/user/ribeirop/audiosetDL/",wavHDF5file = "/om/user/ribeirop/audiosetDL/thousand_tests.hdf5",SR=16000,N=40,SAMPLE_FACTOR=4,LOW_LIM=20,HIGH_LIM=8000,compression='sqrt',ENV_SR=200):
    #create the files for the databases  
    basename = os.path.splitext(os.path.basename(wavHDF5file))[0]
    
    #342000 is how long the flatted thing is
    #np.reshape(arr, (171,2000)) to get the original cochleagram
    
    wavHDF5 = h5py.File(wavHDF5file,'r')
    
    wavSet = wavHDF5["/wav"]
    labelSet = wavHDF5["/labels"]
    
    numberofFiles = wavHDF5.attrs.get("size") 
    
    # TODO: make sure its documented that this runs two cochleagrams at a time -- its possible that if longer ones were used it could only do 1. Alternatively, it could be make more generic to determine the max batch size that fits on the GPU.
    audio = wavSet[0:2]
    end_coch_size = [N*SAMPLE_FACTOR+11,ENV_SR*audio.shape[-1]/SR]
    
    Storage = h5py.File("{0}{1}_Coch.hdf5".format(outputfile,basename))
    if(not "/coch" in Storage):
        cochSet = Storage.create_dataset("coch",(numberofFiles,end_coch_size[0]*end_coch_size[1]),dtype=np.float32)
    else:
        cochSet = Storage["/coch"]
        
    if(not "/labels" in Storage):
        # TODO: change 527 so that it looks at labels and gets the max number -- don't hard code it in. 
        cochlabelSet = Storage.create_dataset("labels",(numberofFiles,527),dtype=bool)
    else:
        cochlabelSet = Storage["/labels"]
    
    Storage.attrs.create("size",numberofFiles)
    
    with tf.Graph().as_default():
        #makes the graph
        if len(audio.shape) == 1: # we need to make sure the input node has a first dimension that corresponds to the batch size
            audio = np.expand_dims(audio,0) 
        nets = {}
        nets['input_signal'] = tf.Variable(audio, dtype=tf.float32)
        nets = tfcochleagram.cochleagram_graph(nets, audio.shape[-1], SR,LOW_LIM=LOW_LIM, HIGH_LIM=HIGH_LIM,N=N,SAMPLE_FACTOR=SAMPLE_FACTOR, compression=compression,ENV_SR=ENV_SR) # use the default values
        nets['cochleagram_reshaped']=tf.reshape(nets['cochleagram'],[2, -1])
        #run_metadata = tf.RunMetadata()
        
        with tf.Session() as sess:
               #LOOP HERE
            start = time.time()
            for i in range(0,numberofFiles-1,2):
             #   print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                audio = wavSet[i:i+2]
                #cochleagram = nets['cochleagram'].eval(feed_dict = {nets['input_signal']:audio})

                cochSet[i:i+2] = nets['cochleagram_reshaped'].eval(feed_dict = {nets['input_signal']:audio})
               
                cochlabelSet[i:i+2] = labelSet[i:i+2]
            
            if(numberofFiles%2 ==1):
                audio = wavSet[numberofFiles-2:numberofFiles]
                cochSet[numberofFiles-2:numberofFiles] = nets['cochleagram_reshaped'].eval(feed_dict = {nets['input_signal']:audio})
                cochlabelSet[numberofFiles-2:numberofFiles] = labelSet[numberofFiles-2:numberofFiles]

            end = time.time()
            
            print("Time ",end-start)
            
            #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            #trace_file = open('timeline.ctf.json', 'w')
            #trace_file.write(trace.generate_chrome_trace_format())


    Storage.close()
    
    MEAN(outputfile,wavHDF5file) #calculates and saves the mean
    print("Done.")


if __name__ == '__main__':
    debug = False
    if debug:
        main()
    else:        
        parser = argparse.ArgumentParser()
        parser.add_argument('-o', '--outputfile') #where to put the hdf5 file
        parser.add_argument('-w', '--wavHDF5file') #where the csv is

        args = vars(parser.parse_args())

        outputfile = args["outputfile"]
        wavHDF5file= args["wavHDF5file"]
        
        main(outputfile = outputfile,wavHDF5file=wavHDF5file)
        
        
