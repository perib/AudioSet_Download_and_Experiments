'''
audiosetDL

Pedro Ribeiro
Josh McDermotts Lab
MSRP BIO 2017

This is a python script for downloading Google AudioSet clips from YouTube.
Dependencies: youtube_dl, ffprobe, ffmpeg

NOTE: This assumes that the start and end times are integer seconds, but google may change this
easy fix by removing the int wraparound in downloadClip()

'''

from __future__ import unicode_literals
import youtube_dl
import os
import subprocess
import time
from multiprocessing import Pool
from multiprocessing import Process
import argparse
import glob

#currentwavset: The set containing all currently downloaded clips
#SectionNumber: The job number for this run.
#ProcessID: The number of this process
#basename: The name of the set (from the csv file)
#errorLog: File to print failed downloads
#goodlog: file to print sucessful downloads
#filename csv file.
#audio_quality: from main(), the quality to download at.
def downloadClip(currentwavset,nextline,SectionNumber,ProcessID,basename, errorLog,goodLog, filename = "ASTenLinks.csv", audio_quality = "44k"):
    #get info:
    #print(ProcessID," " ,SectionNumber, " ", nextline)
    split = nextline.split(",")
    YTID = split[0]
    start = int(float(split[1]))
    end = int(float(split[2]))
    

    #skips if we already downloaded it.
    if("YT_{0}.wav".format(YTID) in currentwavset):
        return

    try:
        
        #downloads the full audio for the youtube video
        downloadMP3 = "youtube-dl http://youtu.be/{0} -q --audio-format 'wav' --extract-audio --audio-quality {1} --output 'YT_{0}tmp.%(ext)s' --restrict-filenames".format(YTID,audio_quality)  
        output = subprocess.check_output(downloadMP3, shell=True)

        #chops the youtube video from the desired time steps
        ChopMP3 = "ffmpeg -loglevel panic -ss {0} -i YT_{1}tmp.wav -t {2} -c:v copy -c:a copy 'YT_{1}.wav'".format(start, YTID, end-start)
        output = subprocess.check_output(ChopMP3, shell=True)
        
        #remove temporary files
        subprocess.check_output("rm YT_{0}tmp.wav".format(YTID), shell=True)

        #it printed correctly, and we should log that
        goodLog.write(nextline)
        
    except:
        #if we can't download the clip, or if there is a problem, remove the temporary files and log it.
        errorLog.write(nextline)
        try:
            subprocess.check_output("rm -f YT_{0}*".format(YTID), shell=True) #remove any remaining files related to this clip
        except:
            pass

    
#ProcessID: from 0 to N, tells us which processs this is
#p_startIndex: tells us where to start reading
#p_sectionSize: Tells us how much to read
#filename: which csv to readfrom 
def runProcess(ProcessID,p_startIndex,p_sectionSize,SectionNumber,filename,basename,audio_quality,currentwavset,multipleFolders = False):
    file = open("../{0}".format(filename),'r')
   
    errorLog = open("../logs/{0}_errorlogs/errorLog_{1}_{2}.csv".format(basename,SectionNumber,ProcessID),'a')
    goodLog = open("../logs/{0}_goodlogs/goodLog_{1}_{2}.csv".format(basename,SectionNumber,ProcessID),'a')


    if(multipleFolders):
        if(not os.path.exists("./{0}_downloads_{1}".format(basename,SectionNumber))):
            try:
                os.makedirs("./{0}_downloads_{1}".format(basename,SectionNumber))
            except:
                pass
        os.chdir("./{0}_downloads_{1}".format(basename,SectionNumber))
    else:
        if(not os.path.exists("./{0}_downloads_0".format(basename))):
            try:
                os.makedirs("./{0}_downloads_0".format(basename))
            except:
                pass
        os.chdir("./{0}_downloads_0".format(basename))
    
    #skip first three lines
    TimeCreated = file.readline()
    nums = file.readline()[9:]
    listofHeaders = file.readline().split(",")
    
    #skip this many
    for _ in range(p_startIndex):
        file.readline()
    #download the next p_sectionSize youtube clips
    for _ in range(p_sectionSize):
        nextline = file.readline()
        downloadClip(currentwavset,nextline,SectionNumber,ProcessID,basename,errorLog,goodLog,filename,audio_quality)
        #print(nextline.split(",")[0])

   
        
#filename: csv file the wav files will be downloaded from
#audio_quality: the quality of audio you want, either in khz or a value between 0 and 9, lower values being better
#numSections: The total number of jobs being submitted
#SectionNumber: Which job this current instance is.
#numprocesses: The number of python multiprocessings to be used per job
#multipleFolders: If True, then each job gets it's own folder, otherwise they all go in <name>_downloads_0
def main(filename = "ASTenLinks.csv", audio_quality = "44k",SectionNumber = 0, numSections = 1, numprocesses = 1,multipleFolders = False ):
    file = open(filename)
    #create the folders where everything will be stored
    basename = os.path.splitext(filename)[0]
    if(not os.path.exists("./{0}_downloads".format(basename))): #stores the downloaded wav files
        try:
            os.makedirs("./{0}_downloads".format(basename))
        except:
            pass
    
    if(not os.path.exists("./logs/{0}_errorlogs".format(basename))): #prints out the files it could not download
        try:
            os.makedirs("./logs/{0}_errorlogs".format(basename))
        except:
            pass
    
    if(not os.path.exists("./logs/{0}_goodlogs".format(basename))): #prints out correctly downloaded files
        try:
            os.makedirs("./logs/{0}_goodlogs".format(basename))
        except:
            pass
    
    os.chdir("./{0}_downloads".format(basename)) #move to the downloads folder

    #skip first three lines which are just information
    TimeCreated = file.readline()
    Total_Number = int(file.readline().split(",")[0][12:]) #count number of links in file
    listofHeaders = file.readline().split(",")
        
    #divide the work between processes
    sectionSize = int(Total_Number/numSections)
    startIndex = sectionSize * (SectionNumber)
    
    p_sectionSize = int(sectionSize/numprocesses)

    wavfiles = glob.glob("./{0}_downloads_*/*.wav".format(basename))
    currentwavset = set() #stores the wav files that have currently been downloaded. 
    for wavpath in wavfiles:
        currentwavset.add(os.path.basename(wavpath))

    #runs the different processes
    processesList = []
    for i in range(0,numprocesses):
        p_startIndex = startIndex + p_sectionSize*(i)
        if(i == numprocesses-1):
            p_sectionSize = p_sectionSize + (sectionSize%numprocesses)
        if(i == numprocesses-1 and SectionNumber == numSections-1):
            p_sectionSize = Total_Number - p_startIndex #Gets the remainder of the list          
        p = Process(target=runProcess,args = (i, p_startIndex,p_sectionSize,SectionNumber,filename,basename,audio_quality,currentwavset,multipleFolders))
        p.start()
        processesList.append(p)   
    #if you want to wait for it to finish
#    for p in processesList:
#        p.join()


def runTest():
    print("start")
    #main(SectionNumber = 0, numSections = 2, numprocesses = 3)
    #main(SectionNumber = 1, numSections = 2, numprocesses = 3)
    numSections = 2 #number of jobs
    numProcesses = 10 #number of python multiprocessing things to run per job
    plist = [] #list of processes so we can wait for them to finish
    startime = time.time()
    for i in range(0,numSections):
        print("main", i)
        p = Process(target=main,args = ("AS23Links.csv", "44k",i, numSections, numProcesses,True))
        p.start()
        plist.append(p)
        #p.join
        #time.sleep(5)

    for p in plist:
        p.join() #wait for all the processes to finish

    end = time.time()
    print("ALL FINISHED")
    print(end-startime)
    
if __name__ == '__main__':
    debug = False #if true, then use the default csv file to read from. 
    if debug:
        runTest()
    else:
        #get args:
        #filename = "ASTenLinks.csv", audio_quality = "44k",SectionNumber = 0, numSections = 1, numprocesses = 1
        
        #filename: csv file the wav files will be downloaded from
        #audio_quality: the quality of audio you want, either in khz or a value between 0 and 9, lower values being better
        #numSections: The total number of jobs being submitted
        #SectionNumber: Which job this current instance is.
        #numprocesses: The number of python multiprocessings to be used per job
        #multipleFolders: If True, then each job gets it's own folder, otherwise they all go in <name>_downloads_0
        
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-a', '--filename')
        parser.add_argument('-b', '--audio_quality')
        parser.add_argument('-c', '--SectionNumber')
        parser.add_argument('-d', '--numSections')
        parser.add_argument('-e', '--numprocesses')
        parser.add_argument('-m', '--multiplefolders')

        args = vars(parser.parse_args())

        filename = args["filename"]
        audio_quality = args["audio_quality"]
        numSections= int(args["numSections"])
        SectionNumber = int(args["SectionNumber"])
        numprocesses  = int(args["numprocesses"])
        multiplefolders = args["multiplefolders"]
        if multiplefolders == "TRUE":
            multiplefolders = True
        else:
            multiplefolders = False

        main(filename,audio_quality,SectionNumber,numSections,numprocesses,multiplefolders)


