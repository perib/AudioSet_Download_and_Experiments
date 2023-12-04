# TODO SAVE PER EPOCH IN NON UNBALANCED SET
import tensorflow as tf
import numpy as np
import h5py
import math
from math import sqrt
import argparse
import os
import random
from PedrosNetworkFunctionsALPHA import *


def main(Learning_Rate, limit=None, music_only=False, print_extras=True, Net_or_VGG="Net", name="", unbalanced=False,
         folder="TB2", Conv1_filtersize=9, TASK="PREDICT_LABELS", OVERIDE_FOLDER=None, SAVE=True, padding="SAME",
         poolmethod="MAXPOOL", conv1_times_hanning=False):
    # if we want to use the unbalanced set for training/eval, read that file in
    if (unbalanced):
        trainfile = "/om/user/ribeirop/audiosetDL/unbalanced_stripped/unbalanced_train_segments_Coch.hdf5"
        train_averagefile = "/om/user/ribeirop/audiosetDL/unbalanced_stripped/unbalanced_train_segments_average.npy"
    else:
        trainfile = "/om/user/ribeirop/audiosetDL/balanced_stripped/balanced_train_segments_Coch.hdf5"
        train_averagefile = "/om/user/ribeirop/audiosetDL/balanced_stripped/balanced_train_segments_average.npy"

    testfile = "/om/user/ribeirop/audiosetDL/eval_stripped/eval_segments_Coch.hdf5"
    test_averagefile = "/om/user/ribeirop/audiosetDL/eval_stripped/eval_segments_average.npy"
    trainset = h5py.File(trainfile, "r")
    testset = h5py.File(testfile, "r")

    train_coch = trainset["/coch"]  # The cochleagrams are in here
    test_coch = testset["/coch"]

    #    unbalanced_label_counts = np.load("/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/unbalanced_label_counts.npy")


    if (TASK == "PREDICT_LABELS"):
        multiple_labels = True
        if (music_only):  # If we want to use a label set without music
            if (unbalanced):
                trainlabels_file = "/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/unbalanced_train_segments_nomusic_labels_only.hdf5"
            else:
                trainlabels_file = "/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/balanced_train_segments_nomusic_labels_only.hdf5"

            testlabels_file = "/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/eval_segments_nomusic_labels_only.hdf5"

            trainlabelsSet = h5py.File(trainlabels_file, "r")
            testlabelsSet = h5py.File(testlabels_file, "r")

            trainlabels = trainlabelsSet["/labels"]
            testlabels = testlabelsSet["/labels"]

            numlabels = 525
        else:
            trainlabels = trainset["/labels"]
            testlabels = testset["/labels"]

            numlabels = 527
    elif (TASK == "PREDICT_COUNTS"):
        multiple_labels = False
        if (unbalanced):
            trainlabels = np.load("/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/unbalanced_label_counts.npy")
        else:
            trainlabels = np.load("/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/balanced_label_counts.npy")

        testlabels = np.load("/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/eval_label_counts.npy")
        numlabels = 15

    COCHLEAGRAM_LENGTH = int(342000)  # full is 342000
    train_mean_coch = np.load(train_averagefile)[0:COCHLEAGRAM_LENGTH]  # mean cochleagram
    # test_mean_coch = np.load(test_averagefile)

    Num_Epochs = 2000

    if (not limit == None):
        trainsize = limit
        testsize = limit
    else:
        trainsize = trainset.attrs.get("size")
        testsize = testset.attrs.get("size")
        limit = "full"

    # for unbalanced set
    cutoff = int(trainsize * .90)

    b = 0
    batchsize = 100

    for epoch_step in range(0, 50):
        epoch_step_folder = "EPOCH_SAVED/{0}/model.ckpt-{1}".format(OVERIDE_FOLDER, epoch_step)

        print(epoch_step_folder)

        print("building graph")
        # The training of the graphs are defined here.
        with tf.Graph().as_default():

            if OVERIDE_FOLDER == None:
                # Get the base graph
                if (Net_or_VGG == "Net"):
                    TensorBoard_Folder_train = './{4}/{0}_{6}_{1}_n{2}_lr{3}_conv1FS{5}_TRAIN'.format(Net_or_VGG, name,
                                                                                                      limit,
                                                                                                      Learning_Rate,
                                                                                                      folder,
                                                                                                      Conv1_filtersize,
                                                                                                      TASK)
                    TensorBoard_Folder_test = './{4}/{0}_{6}_{1}_n{2}_lr{3}_conv1FS{5}_TEST'.format(Net_or_VGG, name,
                                                                                                    limit,
                                                                                                    Learning_Rate,
                                                                                                    folder,
                                                                                                    Conv1_filtersize,
                                                                                                    TASK)
                    Saver_Folder = '/{0}_{5}_{1}_n{2}_lr{3}_conv1FS{4}'.format(Net_or_VGG, name, limit, Learning_Rate,
                                                                               Conv1_filtersize, TASK)
                    nets = Gen_audiosetNet(COCHLEAGRAM_LENGTH, numlabels, train_mean_coch, Conv1_filtersize, padding,
                                           poolmethod, conv1_times_hanning)
                else:
                    TensorBoard_Folder_train = './{4}/{0}_{4}_{1}_n{2}_lr{3}_TRAIN'.format(Net_or_VGG, name, limit,
                                                                                           Learning_Rate, folder, TASK)
                    TensorBoard_Folder_test = './{4}/{0}_{4}_{1}_n{2}_lr{3}_TEST'.format(Net_or_VGG, name, limit,
                                                                                         Learning_Rate, folder, TASK)
                    Saver_Folder = '/{0}_{4}_{1}_n{2}_lr{3}'.format(Net_or_VGG, name, limit, Learning_Rate, TASK)
                    nets = Gen_VGG(COCHLEAGRAM_LENGTH, numlabels, train_mean_coch)
            else:
                TensorBoard_Folder_train = './{0}/{1}_TRAIN'.format(folder, OVERIDE_FOLDER)
                TensorBoard_Folder_test = './{0}/{1}_TEST'.format(folder, OVERIDE_FOLDER)
                Saver_Folder = '/{0}'.format(OVERIDE_FOLDER)
                if (Net_or_VGG == "Net"):
                    nets = Gen_audiosetNet(COCHLEAGRAM_LENGTH, numlabels, train_mean_coch, Conv1_filtersize, padding,
                                           poolmethod, conv1_times_hanning)
                else:
                    nets = Gen_VGG(COCHLEAGRAM_LENGTH, numlabels, train_mean_coch)

            print(TensorBoard_Folder_train)
            print(TensorBoard_Folder_test)
            # Get the loss functions
            Cross_Entropy_Train_on_Labels(nets, numlabels, Learning_Rate, multiple_labels)

            merged = tf.summary.merge_all()

            # CALC MISSING CROSS ENTROPY



            saver = tf.train.Saver()
            with tf.Session() as sess:
                
                saver.restore(sess, epoch_step_folder)
                #sess.run(tf.global_variables_initializer())
                #sess.run(tf.initialize_all_variables())
                #sess.run(tf.initialize_local_variables()) 
                offset = 0  # random.randint(0,100*2000)
                train_writer = tf.summary.FileWriter(
                    TensorBoard_Folder_train)  # tensorboard writters for both train and test accuracy
                test_writer = tf.summary.FileWriter(TensorBoard_Folder_test)

                # EVALUATION
                print("eval")
                # pretty much the same thing but no train step.
                # print("eval")
                # TEST SET
                startindex = cutoff + offset  # start at the first item in the last 10%
                endindex = startindex + batchsize
                xe_sum = 0
                print(startindex)
                numbatches = 0
                calculator = AveragePrecisionCalculator()
                MAPsum = 0
                
                sess.run(tf.initialize_local_variables())
                for v in tf.local_variables():
                    print(v)
                
                #varlist = ['AUC/auc_accumulator/true_positives:0','AUC/auc_accumulator/false_negatives:0','AUC/auc_accumulator/true_negatives:0','AUC/auc_accumulator/false_positives:0']     
                #sess.run(tf.variables_initializer('AUC/auc_accumulator/true_positives:0') )              
                
                print("init auc ", sess.run(nets["auc"]))
                while (endindex <= cutoff + 100 * batchsize + offset):  # trainsize):
                    numbatches = numbatches + 1  # TODO: just calculate this
                    # print(startindex)
                    if (numbatches % 10 == 0):
                        print(numbatches)

                    trainlabelbatch = trainlabels[startindex:endindex]
                    if TASK == "PREDICT_LABELS":
                        predictions = sess.run(nets['predicted_labels'], feed_dict={
                            nets['input_to_net']: train_coch[startindex:endindex][:, 0:COCHLEAGRAM_LENGTH],
                            nets['actual_labels']: trainlabelbatch, nets['keep_prob']: 1})
                    else:
                        predictions = sess.run(nets['predicted_labels'], feed_dict={
                            nets['input_to_net']: train_coch[startindex:endindex][:, 0:COCHLEAGRAM_LENGTH],
                            nets['actual_labels']: trainlabelbatch, nets['keep_prob']: 1})

                    for i in range(len(predictions)):
                        # print(predictions[i])
                        # print(trainlabelbatch[i])
                        normed = predictions[i]  # / abs(predictions[i]).max()
                        sort = normed.argsort()[::-1]
                        p = normed[sort][0:15]
                        a=  trainlabelbatch[i][sort][0:15]
                        c = len(np.where(trainlabelbatch[i] > 0)[0])
                        calculator.accumulate(p,a,c)
                        MAPsum = MAPsum + AveragePrecisionCalculator.ap_at_n(p,a,15,c)
                        # calculator.accumulate(np.random.rand(527),trainlabelbatch[i])
                        
                    aucp = (predictions- predictions.min(axis=0))/(predictions.max(axis=0)-predictions.min(axis=0))
                    sess.run(nets['update_auc'],feed_dict={nets['l']:trainlabelbatch,nets['p']:aucp})
    
                        
                    startindex = endindex
                    endindex = startindex + batchsize

                # print(nets['global_step'].eval(session=sess))
                print("MAP ", MAPsum / numbatches)
                MAPsummary = sess.run(nets['MAPsum'], feed_dict={nets['MAP']: MAPsum / numbatches})
                test_writer.add_summary(MAPsummary, nets['global_step'].eval(session=sess))

                
                print("GAP ",calculator.peek_ap_at_n())
                GAPsummary = sess.run(nets['GAPsum'], feed_dict={nets['GAP']: calculator.peek_ap_at_n()})
                test_writer.add_summary(GAPsummary, nets['global_step'].eval(session=sess))
                
                print("AUC " ,sess.run(nets["auc"]))
                
                test_writer.add_summary(sess.run(nets['AUCsum']), nets['global_step'].eval(session=sess))
                

                # TRAIN
                print("train")
                offset = 0  # random.randint(0,cutoff - 200*100)
                startindex = 0 + offset
                endindex = startindex + batchsize
                
                
                xe_sum = 0
                
                numbatches = 0
                calculator = AveragePrecisionCalculator()
                MAPsum = 0 #sum of the individuals APs
                sess.run(tf.initialize_local_variables())
                for v in tf.local_variables():
                    print(v)
                    
                print("init auc ", sess.run(nets["auc"]))
                # loop through the first 90% of the set
                while (endindex <= batchsize * 100 + offset):
                    # Get the indeces as ints for the labels
                    numbatches = numbatches + 1  # TODO: just calculate this
                    trainlabelbatch = trainlabels[startindex:endindex]
                    if (numbatches % 10 == 0):
                        print(numbatches)
                    if TASK == "PREDICT_LABELS":
                        predictions = sess.run(nets['predicted_labels'], feed_dict={
                            nets['input_to_net']: train_coch[startindex:endindex][:, 0:COCHLEAGRAM_LENGTH],
                            nets['actual_labels']: trainlabelbatch, nets['keep_prob']: 1})
                    else:
                        predictions = sess.run(nets['predicted_labels'], feed_dict={
                            nets['input_to_net']: train_coch[startindex:endindex][:, 0:COCHLEAGRAM_LENGTH],
                            nets['actual_labels']: trainlabelbatch, nets['keep_prob']: 1})

                        #predictions is the output of the last layer, that a list of activations per 527 labels
                    
                    for i in range(len(predictions)): #loop through the predictions for each sample in the batch
                        normed = predictions[i]  # / abs(predictions[i]).max()
                        sort = normed.argsort()[::-1] #sort from greatest to smallest
                        p = normed[sort][0:15] #take the 15 biggest
                        a=  trainlabelbatch[i][sort][0:15] #the label corresponding to the top 15 predictions (1 if present, 0 otherwise.)
                        c = len(np.where(trainlabelbatch[i] > 0)[0]) #how many labels exist total.
                        calculator.accumulate(p,a,c) #calculates GAP automatically
                        MAPsum = MAPsum + AveragePrecisionCalculator.ap_at_n(p,a,15,c) #calculates precision of the point and adds it in.
                    
                    aucp = (predictions- predictions.min(axis=0))/(predictions.max(axis=0)-predictions.min(axis=0)) #normalize values to between 0 and 1 to calculate AUC
                    sess.run(nets['update_auc'],feed_dict={nets['l']:trainlabelbatch,nets['p']:aucp}) #calculates AUC with tensorflows function.
        
                        
                    startindex = endindex
                    endindex = startindex + batchsize

                #gets final values and saves
                print(calculator.peek_ap_at_n())
                GAPsummary = sess.run(nets['GAPsum'], feed_dict={nets['GAP']: calculator.peek_ap_at_n()})
                train_writer.add_summary(GAPsummary, nets['global_step'].eval(session=sess))
                
                print(MAPsum / numbatches)
                MAPsummary = sess.run(nets['MAPsum'], feed_dict={nets['MAP']: MAPsum / numbatches*100})
                train_writer.add_summary(MAPsummary, nets['global_step'].eval(session=sess))
                
                print("AUC " ,sess.run(nets["auc"]))
                train_writer.add_summary(sess.run(nets['AUCsum']), nets['global_step'].eval(session=sess))
                
               
                
    print("done")


if __name__ == '__main__':
    debug = False  # if true, then use the default csv file to read from.
    if debug:
        learning_rate = 1e-4
        music_only = False  # currently also removing speech
        limit = None
        Net_or_VGG = "Net"
        name = "BBB_TEST5"
        unbalanced = True
        folder = "TB_GMAP"
        Conv1_filtersize = 9
        TASK = "PREDICT_LABELS"
        # TASK = "PREDICT_COUNTS"
        OVERIDE_FOLDER = "Net_PREDICT_LABELS_A2UN91_4_HP9_nfull_lr0.0001_conv1FS9"
        # OVERIDE_FOLDER =  "Net_UNBALANCED91_5_nfull_lr1e-05"
        SAVE = False
        padding = "SAME"
        # padding = "VALID"
        poolmethod = "HPOOL"
        # poolmethod = "MAXPOOL"
        conv1_times_hanning = False
        main(learning_rate, limit, music_only, Net_or_VGG=Net_or_VGG, name=name, unbalanced=unbalanced, folder=folder,
             Conv1_filtersize=Conv1_filtersize, TASK=TASK, OVERIDE_FOLDER=OVERIDE_FOLDER, SAVE=SAVE, padding=padding,
             poolmethod=poolmethod, conv1_times_hanning=conv1_times_hanning)

    else:

        parser = argparse.ArgumentParser()
        parser.add_argument('-l', '--learning_rate' , default = 1e-5)
        parser.add_argument('-z', '--limit', default = None)
        parser.add_argument('-M', '--Model', default = "Net")
        parser.add_argument('-n', '--name', default = "noname")
        parser.add_argument('-f', '--conv1filtersize', default = 9)
        parser.add_argument('-t', '--TASK',default = "PREDICT_LABELS")
        parser.add_argument('-o', '--OVERIDE_FOLDER', default = None)
        parser.add_argument('-p', '--poolmethod', default = "MAXPOOL")
        parser.add_argument('-b', '--tbfolder', default = "TB2")
        parser.add_argument('-m', '--musiconly', default = None)
        parser.add_argument('-x', '--conv1_times_hanning', default = None)
        parser.add_argument('-e', '--Freeze_Model_File', default = None)
        parser.add_argument('-s', '--save', default = None)
        
        args = vars(parser.parse_args())

        Freeze_Model_File = args["Freeze_Model_File"]
        
        learning_rate = float(args["learning_rate"])
        
        if (not args["limit"] == None):
            limit = int(args["limit"])
        else:
            limit = None
        
        Net_or_VGG = args["Model"]
        name = args['name']
        Conv1_filtersize = int(args['conv1filtersize'])
        TASK = args['TASK']
        
        if (not args['OVERIDE_FOLDER'] == None):
            OVERIDE_FOLDER = args['OVERIDE_FOLDER']
        else:
            OVERIDE_FOLDER = None
        
        if (not args['musiconly'] == None):
            if (args['musiconly'] == "True"):
                music_only = True
            else:
                music_only = False
        else:
            music_only = False
        
        
        if (not args['conv1_times_hanning'] == None):
            if (args['conv1_times_hanning'] == "True"):
                conv1_times_hanning = True
            else:
                conv1_times_hanning = False
        else:
            conv1_times_hanning = False
            
            
        if (not args['save'] == None):
            if (args['save'] == "True"):
                SAVE = True
            else:
                SAVE = False
        else:
            SAVE = True
        
        unbalanced = True
        
        folder =  args['tbfolder']
        poolmethod =args['poolmethod']  
        
        padding = "SAME"
        main(learning_rate, limit, music_only, Net_or_VGG=Net_or_VGG, name=name, unbalanced=unbalanced, folder=folder, Conv1_filtersize=Conv1_filtersize, TASK=TASK, OVERIDE_FOLDER=OVERIDE_FOLDER, SAVE=SAVE, padding=padding, poolmethod=poolmethod, conv1_times_hanning=conv1_times_hanning)