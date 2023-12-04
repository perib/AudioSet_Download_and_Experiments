# AudioSetDL
A set of python scripts that download [Google AudioSet](https://research.google.com/audioset/index.html) audio clips from Youtube, packages Wav files into HDF5, then converts them into cochleagrams stored in another HDF5 file.

**Data**
Google released three CSV files found on [this page](https://research.google.com/audioset/download.html). These CSV files are read in by the program to fetch the links, timestamps, and labels for each audio sample.

>The dataset is divided in three disjoint sets: a balanced evaluation set, a balanced training set, and an unbalanced training set. In the balanced evaluation and training sets, we strived for each class to have the same number of examples. The unbalanced training set contains the remainder of annotated segments.
Evaluation - eval_segments.csv
20,383 segments from distinct videos, providing at least 59 examples for each of the 527 sound classes that are used. Because of label co-occurrence, many classes have more examples.
Balanced train - balanced_train_segments.csv
22,176 segments from distinct videos chosen with the same criteria: providing at least 59 examples per class with the fewest number of total segments.
Unbalanced train - unbalanced_train_segments.csv
2,042,985 segments from distinct videos, representing the remainder of the dataset.

[class_labels_indices.csv](http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv) contains the mappings from the labels to english labels and their indices for one-hot.


## Dependencies:

scipy

numpy

[Youtube_dl](https://github.com/rg3/youtube-dl/blob/master/README.md#readme)

[ffprob/ffmpeg](https://ffmpeg.org/ffprobe.html) (this is readily available in the cluster as `openmind/ffmpeg/20160310`)

[HDF5/H5py](http://www.h5py.org/)

[Tensorflow](https://www.tensorflow.org/)

[tensorflow cochleagram](https://github.mit.edu/jfeather/tfcochleagram)

[python cochleagram stuff](https://github.mit.edu/raygon/py-cochleagram) (required for tfcochleagram):




## USE

Here I list the programs in the order they should be run, and how to use them.
There are two versions of each script, a jupyter notebook ending in <name>.ipyno, and a python file ending in <name>_python.py. The jupyter notebook is helpful for running interactively or debugging, but the python file should be called when downloading a larger set.

### audiosetDL.<>

This function does the actual downloading of the audio files. It will skip any wav files already found in the directory, and thus of stopped, can continue downloading from where it left off. Several videos may fail to download due to connection errors, but rerunning the program once or twice should catch these. You can verify whether a connection error occurred by looking in the error .err output of the job for an 'error 104, connection reset by peer' exception. Some other videos may fail to download due to differences in country availability, privacy settings, deletion of video itself, or a wide number of things. Many of these are simply gone and impossible to retrieve, so it is not possible to download the entirety of the dataset.
It creates a couple of folders to store different things. The <name> of the set is taken from the csv file, (as in <name>.csv)

**./<name>_downloads/<name>_downloads_X:** where x is an int 0 or greater. This stores the actual Wav files. The nested folder is numbered according to the job, with each job having it's own folder or all jobs saving to <name>_downloads_0 depending on the flags set.

The wav files will be named `YT_<YTID>.wav` where YTID is the youtube ID, which is a unique string that is linked to a particular video. 

NOTE: Several temporary files will be saved her during downloading/chopping of the audio clips. These often end in .ytdl, .m4p, .frag, <...>tmp.wav, etc. However they all have "tmp" in the name, and can thus be removed by calling `rm -r *tmp*`. This will remove all temporary files, but may also remove some actual completed wavs. However this is unlikely and it is very easy to redownload, so this is generally the easiest method if you are planning on rerunning the program anyway.

**./logs/<name>_errorlogs** : This stores a csv per job that lists the links in which it failed to download.

**./logs/<name>_goodlogs** : This stores a csv per job that lists the links in which it sucessfully downloaded.


**Arguments**

'-a', '--filename': csv file the wav files will be downloaded from

'-b', '--audio_quality': the quality of audio you want, either in khz or a value between 0 and 9, lower values being better

'-c', '--SectionNumber': The total number of jobs being submitted

'-d', '--numSections' Which job this current instance is.

'-e', '--numprocesses': The number of python multiprocessings to be used per job

'-m', '--multiplefolders': If True, then each job gets it's own folder, otherwise they all go in <name>_downloads_0

An example of how to correctly run this program is found in Downloads.sh


**known issues**

If Google updates the csv file, this program currently will not remove/update any existing files.

Several files are unavailable to download for one reason or another, so there's not a way to get the complete set. 

Connection errors sometimes cause a download to fail, but this is easily sovled by rerunning the program. 


### Wav_into_HDF5L.<>

This script simply goes through all the files in the downloads folders and converts them into a single HDF5 file. All wav files are downsampled to 16K when read in. If the file is not 10 seconds long, 0 are padded in before downsampling. Then all files are converted into mono if in stereo. This yields an 16k * 10 or 160K length array of floats. Finally, the array is normalized by dividing by the largest absolute value in the array. 

The HDF5 file is named <name>.hdf5 and stores the normalized mono-ized wav files as well as the one-hot labeling taken from the csv file.

in addition metadata is stored in a numpy object called <name>_metadata.npy where each index corresponds to the same index in the HDF5 file. This stores the YTID, start time, end time, labels, normalization factor, and whether or not the clip needed to be padded. This information can be indexed through 'YTID', 'start', 'end', 'labels', 'norm', amd 'padded' respectfully.

The HDF5 file has two directories:

** /wav ** : (N,160000) stores the wav files as a 160K length array.
** /labels **: (N, 527) stores one-hot multiclass labels as booleans, with indices corresponding to the label.

There is also one attribute on the dataset itself, "size" is the total number of wavs in this set.
NOTE: For whatever reason, either due to corruption or other weird cases, a very small number of wavs may not be read in. Since the size of the hdf5 file is initialized assuming all files could be read, it's shape will be larger than the number of actual files in it. Thus, when referencing the number of files, use the size attribute and not the shape.

**Arguments**

'-p', '--downloadPath' : the path to the outer downloads folder. (as in the one that holds the <name>_downloads_X folders)
'-o', '--outputPath' : path to where you want the final HDF5/numpy object file to be stored.
'-c', '--csv' : path to the csv file

An example of how to correctly run this program is found in WavToHDf5.sh.

### WavHDF5_to_CochHDF5.<>
This program takes the HDF5 file stored above to generate and save cochleagrams from the wavs. The cochleagram is then flattened to make the HDF5 file more efficient. To unflatten, you can simply call `np.reshape(arr, (171,2000))`

The HDF5 file will be saved as <name>_Coch.hdf5
The average cochleagram will also be saved as a numpy object in <name>_average.npy

The HDF5 file contains two directories:

** /coch ** : (N, 342000) stores the flattened cochleagram.
** /labels **: (N, 527) stores one-hot multiclass labels as booleans, with indices corresponding to the label. Copied exactly as the above.

There is also one attribute on the dataset itself, "size" is the total number of wavs in this set.
NOTE: For whatever reason, either due to corruption or other weird cases, a very small number of wavs may not be read in. Since the size of the hdf5 file is initialized assuming all files could be read, it's shape will be larger than the number of actual files in it. Thus, when referencing the number of files, use the size attribute and not the shape.


**Arguments**

'-o', '--outputPath' : path to where you want the final HDF5 file to be stored.
'-w', '--wavHDF5file' : path to the HDF5 file containing the wavs created by Wav_into_HDF5L


An example of how to correctly run this program is found WavToCoch.sh.
