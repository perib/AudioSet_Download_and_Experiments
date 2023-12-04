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
import time

#creates a dictionary to go from youtube links to labels
def genDictionary(path):
    f = open(path)
    dic = {}
    f.readline()
    f.readline()
    f.readline()
    
    for line in reader(f,quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        base = line[0]
        dic[base] = line
    
    return dic

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

#loads the audio file, also padds if needed
def load_audio_wav(audio_path, DUR = 10, resample_SR = 16000, START = 0):
    # audio_path should be a .wav file (including the extension)
    # resamples the audio to the desired rate
    # Cuts audio to be only a particular duration
    # Starts at START in case you want laterin the file. 
    SR, audio = scipy.io.wavfile.read(audio_path)
    #check if shorter than 10 seconds and pad zeroes to the rest
    padded = False
    
    if(len(audio.shape) == 1): #is mono
        if(len(audio) < SR * 10):
            pad = (SR * 10) - len(audio)
            audio = np.pad(audio,(0,pad),'constant')
            padded = True
    else: #is stereo
        if(audio.shape[0] < SR * 10):
            pad = (SR*10) - audio.shape[0]
            audio = np.pad(audio,[(0,pad),(0,0)],'constant')
            padded = True

    audio = audio[START:START + SR*DUR]
    if SR != resample_SR:
        audio = signal.resample(audio, resample_SR*DUR) # in actuality this might introduce artifacts, probably want to resample and then cut out the section we want. 
        SR = resample_SR    
    audio = audio[START:START + SR*DUR]
    return audio, resample_SR,padded 

#the maximum absolute value
def maxabsolute(nparray):
    return abs(max(np.amax(nparray),np.amin(nparray),key=abs))

#turns stereo into mono and normalizes the array
#if the audio is all zeroes, just return the array and 0
def Make_mono(audio):
    audio = audio.astype(np.float32)
    if(len(audio.shape) == 2): 
        mono = audio.sum(axis=1) / 2.0
        absmax = maxabsolute(mono)
        if(absmax ==0):
            return mono, absmax
        mono = mono/absmax
        return mono, absmax
    else:
        audio = audio.astype(np.float32)
        absmax = maxabsolute(audio)
        if(absmax ==0):
            return audio, absmax
        audio = audio/absmax
        return audio, absmax

#outputfile: where to save the hdf5 file
#csvfile: where the csv file is
def main(outputfile = "/om/user/ribeirop/audiosetDL/",csvfile = "/home/ribeirop/OMFOLDER/audiosetDL/thousand_tests.csv",Path ="/home/ribeirop/OMFOLDER/audiosetDL/thousand_tests_downloads/"):
    #create the files for the databases
    basename = os.path.basename(os.path.splitext(csvfile)[0])
    if(Path == None):
        Path = "./{0}_downloads".format(basename)
     
    wavfiles = glob.glob("{0}/**/*.wav".format(Path)) #gets a list of all the downloaded wav files

    indeces_dict = genLabelDictionary("class_labels_indices.csv")
    dictionary = genDictionary(csvfile)
    
    np.random.shuffle(wavfiles) #randomize the list so that we can get random samples more easily
    numfiles = len(wavfiles)
    
    Storage = h5py.File("{0}{1}.hdf5".format(outputfile,basename))
    if(not "/wav" in Storage):
        wavSet = Storage.create_dataset("wav",(numfiles,160000),dtype=np.float32)
    else:
        wavSet = Storage["/wav"]
        
    if(not "/labels" in Storage):
        labelSet = Storage.create_dataset("labels",(numfiles,527),dtype=bool)
    else:
        labelSet = Storage["/labels"]

    #to create and save the numpy metadata objext
    names = ["YTID","start","end","labels","norm","padded"]
    formats = ["|S20",'i4','i4','|S1000','f8','b']
    dtype = dict(names = names, formats=formats)
    metaarr = []
    
    starttime = time.time()
    
    #index of where the current audio sample will go to.
    index = 0 #in case we need to skip one for whatever reason.
    for wav in wavfiles:
        try:
            filepath = wav
            audio, resample_SR, padded = load_audio_wav(filepath) #SR is sampling rate
            mono, normalizingNumber = Make_mono(audio)

            YTID = os.path.splitext(os.path.basename(wav))[0][3:]
            data = dictionary[YTID]
            start = int(float(data[1]))
            end = int(float(data[2]))
            labels = data[3]
            
            #SANITY CHECK:
            if(YTID == None or start == None or end == None or labels == None or normalizingNumber == None or padded == None):
                print("NONETYPE ERROR: {0}".format(YTID) )
                
            metadata = (YTID,start,end,labels,normalizingNumber,padded)

            #creates one-hot labeling
            indeces = [False]*527
            for label in labels.split(","):
                indeces[int(indeces_dict[label])] = True
            wavSet[index] = mono
            metaarr.append(metadata)
            labelSet[index] = indeces
            index = index + 1
        except Exception as inst:
            print("ERROR READING: {0}".format(wav)) #shouldn't happen
            print(inst)
    
    Storage.attrs.create("size",index)
            
    endtime = time.time()
    print("time")
    print(endtime-starttime)
            
    np.save("{0}{1}_metadata".format(outputfile,basename),np.array(metaarr,dtype=dtype))
            
    
    print("done")
    
    

            



if __name__ == '__main__':
    debug = False
    if debug:
        main()
    else:        
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--downloadPath') #where the downloads happened
        parser.add_argument('-o', '--outputPath') #where to put the hdf5 file
        parser.add_argument('-c', '--csv') #where the csv is

        args = vars(parser.parse_args())

        downloadPath= args["downloadPath"]
        outputPath= args["outputPath"]
        csvfile = args["csv"]
        
        main(Path = downloadPath, outputfile = outputPath,csvfile=csvfile)
        
        
