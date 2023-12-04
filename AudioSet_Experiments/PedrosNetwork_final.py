#TODO SAVE PER EPOCH IN NON UNBALANCED SET
import tensorflow as tf
import numpy as np
import h5py
import math
from math import sqrt
import argparse
import os
from PedrosNetworkFunctions import *

#This trainer runs its evalutations on the evaluation set and balanced set separetely after training
def train1(test_coch,testlabels,testsize,Saver_Folder,TensorBoard_Folder_train,TensorBoard_Folder_test,batchsize,Num_Epochs,nets,trainsize,train_coch,trainlabels,COCHLEAGRAM_LENGTH):
    
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)

    sv = tf.train.Supervisor(logdir="./Saved_Sessions{0}".format(Saver_Folder),saver = saver, summary_op=None, save_model_secs = 7200,checkpoint_basename='model.ckpt')
    #with tf.Session() as sess:
    with sv.managed_session() as sess:
        #sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(TensorBoard_Folder_train,sess.graph)
        test_writer = tf.summary.FileWriter(TensorBoard_Folder_test,sess.graph)

        #nets['coch_images']
        startindex = 0
        endindex = startindex+batchsize
        '''if(print_extras):
            coch_images_sum = sess.run(nets['coch_images'],feed_dict={nets['input_to_net']:test_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH]})
            train_writer.add_summary(coch_images_sum, 0)  '''  

        #LOOP THROUGH EPOCHS
        print("starting")
        for epoch in range(sess.run(nets['current_epoch']),Num_Epochs):
            if(sv.should_stop()):
                break
                
            image_conv1 = sess.run(nets['conv1_weight_image'])
            train_writer.add_summary(image_conv1, epoch)
            #print("epoch: ",epoch)
            #Train a full epoch
            startindex = sess.run(nets['current_step']) * batchsize
            endindex = startindex+batchsize

            while(endindex <= trainsize):
                '''if(print_extras):
                    _,cross_entropySummary,l1Sums,__ = sess.run([nets['train_step'],nets['cross_entropy_summary'],nets['layer1'],nets['increment_current_step']],feed_dict={nets['input_to_net']:train_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:trainlabels[startindex:endindex],nets['keep_prob']:.5})
                    train_writer.add_summary(cross_entropySummary, epoch)
                    train_writer.add_summary(l1Sums, epoch)
                else:'''

                sess.run([nets['train_step'],nets['increment_current_step']],feed_dict={nets['input_to_net']:train_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:trainlabels[startindex:endindex],nets['keep_prob']:.5})

                startindex = endindex
                endindex = startindex+batchsize

      #EVALUATION!!!      
            if(epoch%1 == 0):
                #print("eval")
                #max_to_keep
                #keep_checkpoint_every_n_hours
                #Evaluate the full dataset
                ###
                #TEST SET
                startindex = 0
                endindex = startindex+batchsize
                total_count = 0
                correct = 0
                correct_train = 0
                while(endindex <= testsize):
                    trainBatchLabels = testlabels[startindex:endindex]
                    indeces_list = []
                    for row in trainBatchLabels:
                        indeces= np.where(row==1)[0]
                        total_count = total_count + len(indeces)
                        indeces_list.append(np.pad(indeces,[0,15-len(indeces)],mode = 'constant',constant_values=-1))

                    '''if(print_extras):
                        addme,pred_sum  = sess.run([nets['numCorrect'],prediction_sums],feed_dict={nets['input_to_net']:test_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:testlabels[startindex:endindex],nets['label_indeces']:indeces_list,nets['keep_prob']:1})
                        train_writer.add_summary(pred_sum, epoch + startindex/testsize)
                        correct = correct + addme
                    else:'''

                    addme  = sess.run(nets['numCorrect'],feed_dict={nets['input_to_net']:test_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:testlabels[startindex:endindex],nets['label_indeces']:indeces_list,nets['keep_prob']:1})
                    correct = correct + addme    


                    '''if(print_extras):
                        if(startindex == 0):
                            conv1array,conv1weights,conv1s = sess.run([nets['conv1'],nets['conv1_Weights'],nets['conv1_sums']],feed_dict={nets['input_to_net']:test_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:testlabels[startindex:endindex],nets['label_indeces']:indeces_list,nets['keep_prob']:1})
                            train_writer.add_summary(conv1s)'''


                    startindex = endindex
                    endindex = startindex+batchsize

                #print("Epoch {0}, accuracy {1}".format(epoch,correct/total_count))
                summary = sess.run(nets['accsum'],feed_dict={nets['accuracy']:correct/total_count})
                test_writer.add_summary(summary, epoch)


                #TRAINSET
                startindex = 0
                endindex = startindex+batchsize
                total_count = 0
                correct = 0
                while(endindex <= trainsize):
                    trainBatchLabels = trainlabels[startindex:endindex]
                    indeces_list = []
                    for row in trainBatchLabels:
                        indeces= np.where(row==1)[0]
                        total_count = total_count + len(indeces)
                        indeces_list.append(np.pad(indeces,[0,15-len(indeces)],mode = 'constant',constant_values=-1))

                    addme  = sess.run(nets['numCorrect'],feed_dict={nets['input_to_net']:train_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:trainlabels[startindex:endindex],nets['label_indeces']:indeces_list,nets['keep_prob']:1})
                    correct = correct + addme    
                    startindex = endindex
                    endindex = startindex+batchsize

                #print("Epoch {0}, accuracy {1}".format(epoch,correct/total_count))
                summary = sess.run(nets['accsum'],feed_dict={nets['accuracy']:correct/total_count})
                train_writer.add_summary(summary, epoch)





            sess.run(nets['increment_current_epoch'])
            sess.run(nets['reset_current_step'])
            sv.saver.save(sess,'./Saved_Sessions{0}/model.ckpt'.format(Saver_Folder))

            '''from tensorflow.python.client import timeline
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            trace_file = open('timeline.ctf.json', 'w')
            trace_file.write(trace.generate_chrome_trace_format())'''
            #endindex = startindex+batchsize
            #potato = sess.run(nets['indeces'],feed_dict={nets['input_to_net']:test_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:testlabels[startindex:endindex],nets['label_indeces']:indeces_list,nets['keep_prob']:1})
            #print()
            #print(start)
            #print(end)
            #print(total_count)
            #print(correct)
            #print("actual: ", indeces_list)
            #print("predicted :", potato)
            #print()
            #f= open("output3.csv","a")
            #f.write("loop {0}, accuracy {1}\n".format(i,correct/total_count))
                #f.close()''

                
#runs evaluation on the trainset itself. Splits training set 90 to 10
def train2(test_coch,testlabels,testsize,Saver_Folder,TensorBoard_Folder_train,TensorBoard_Folder_test,batchsize,Num_Epochs,nets,trainsize,train_coch,trainlabels,COCHLEAGRAM_LENGTH,cutoff,TASK,SAVE = True):
    #This saver saves the graph at every epoch, and never deletes/uses them. (otherwise the superviser will delete old saves)
    epochsaver = tf.train.Saver(max_to_keep=0) #saves all the epochs
    if(not os.path.exists("EPOCH_SAVED{0}/".format(Saver_Folder))):
        try:
            os.makedirs("EPOCH_SAVED{0}/".format(Saver_Folder))
        except:
            pass
    
    if SAVE:
        save_model_secs = 7200
        keep_checkpoint_every_n_hours = 5
    else:
        save_model_secs = 0
        keep_checkpoint_every_n_hours =0
    
    #This saver saves every 5 hours but only keeps the latest 5. for backups in case we crash
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)
    sv = tf.train.Supervisor(logdir="./Saved_Sessions{0}".format(Saver_Folder),saver = saver, summary_op=None, save_model_secs = save_model_secs,checkpoint_basename='model.ckpt')
    
    print("initializing")
    with sv.managed_session() as sess: #runs a supervised session
        train_writer = tf.summary.FileWriter(TensorBoard_Folder_train,sess.graph) #tensorboard writters for both train and test accuracy
        test_writer = tf.summary.FileWriter(TensorBoard_Folder_test,sess.graph)

        startindex = 0
        endindex = startindex+batchsize

        #LOOP THROUGH EPOCHS
        #print("starting unbalanced")
        print("starting")
        for epoch in range(sess.run(nets['current_epoch']),Num_Epochs):
            if(sv.should_stop()):
                break
            print("epoch ", epoch)
            
            if SAVE:
                sv.saver.save(sess, 'Saved_Sessions{0}/model.ckpt'.format(Saver_Folder), global_step=epoch)
                epochsaver.save(sess, save_path="EPOCH_SAVED{0}/model.ckpt".format(Saver_Folder), global_step=epoch)
            
            image_conv1 = sess.run(nets['conv1_weight_image'])
            train_writer.add_summary(image_conv1,nets['global_step'].eval(session=sess))
            #train_writer.add_summary(image_conv1,epoch)
            #print("epoch: ",epoch)
            #Train a full epoch
            
            startindex = sess.run(nets['current_step']) * batchsize #continue were we left off if we crashed
            endindex = startindex+batchsize

            #loop through the first 90% of the set
            while(endindex <= cutoff):
                #Get the indeces as ints for the labels 
                trainBatchLabels = trainlabels[startindex:endindex]
                total_count = 0
                correct = 0
                correct_train = 0
                indeces_list = []
                for row in trainBatchLabels:
                    indeces= np.where(row==1)[0]
                    total_count = total_count + len(indeces)
                    indeces_list.append(np.pad(indeces,[0,15-len(indeces)],mode = 'constant',constant_values=-1))

                 #run the train step, increment the current step we are in, calculate how many correct predictions           
                if TASK == "PREDICT_LABELS":
                    _,__,correct,XE_SUM = sess.run([nets['train_step'],nets['increment_current_step'],nets['numCorrect'],nets['cross_entropy_summary']],feed_dict={nets['input_to_net']:train_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:trainlabels[startindex:endindex],nets['label_indeces']:indeces_list,nets['keep_prob']:.5})
                else:
                    _,__,correct,XE_SUM = sess.run([nets['train_step'],nets['increment_current_step'],nets['numCorrect_top_pred'],nets['cross_entropy_summary']],feed_dict={nets['input_to_net']:train_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:trainlabels[startindex:endindex],nets['label_indeces']:indeces_list,nets['keep_prob']:.5})
                
                #write to tensorboard
                
                summary = sess.run(nets['accsum'],feed_dict={nets['accuracy']:correct/total_count})
                #print("epoch {0}, {1}".format(epoch,correct/total_count))
                '''print(sess.run([nets['Top_1_index'],nets['actual_labels'],nets['predicted_labels']],feed_dict={nets['input_to_net']:train_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:trainlabels[startindex:endindex],nets['label_indeces']:indeces_list,nets['keep_prob']:.5}))
                print(np.array(indeces_list)[:,0])
                print(total_count)
                print(correct)
                print()'''
                train_writer.add_summary(summary,nets['global_step'].eval(session=sess))
                train_writer.add_summary(XE_SUM,nets['global_step'].eval(session=sess))
                
                
                #train_writer.add_summary(summary,epoch +startindex/cutoff)
                #print(epoch +startindex/cutoff)
                startindex = endindex
                endindex = startindex+batchsize
        
                
      #EVALUATION
        #pretty much the same thing but no train step.
            if(epoch%1 == 0):
                #print("eval")
                #TEST SET
                startindex = cutoff #start at the first item in the last 10%
                endindex = startindex+batchsize
                total_count = 0
                correct = 0
                correct_train = 0
                xe_sum = 0
                
                numbatches = 0
                while(endindex <= trainsize):
                    numbatches = numbatches + 1 #TODO: just calculate this
                    #print(startindex)
                    trainBatchLabels = trainlabels[startindex:endindex]
                    indeces_list = []
                    for row in trainBatchLabels:
                        indeces= np.where(row==1)[0]
                        total_count = total_count + len(indeces)
                        indeces_list.append(np.pad(indeces,[0,15-len(indeces)],mode = 'constant',constant_values=-1))
                    
                                        
                    if TASK == "PREDICT_LABELS":
                        addme, xe  = sess.run([nets['numCorrect'],nets['cross_entroy']],feed_dict={nets['input_to_net']:train_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:trainlabels[startindex:endindex],nets['label_indeces']:indeces_list,nets['keep_prob']:1})
                    else:
                        addme, xe   = sess.run([nets['numCorrect_top_pred'],nets['cross_entroy']],feed_dict={nets['input_to_net']:train_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:trainlabels[startindex:endindex],nets['label_indeces']:indeces_list,nets['keep_prob']:1})
                   
                    correct = correct + addme    
                    xe_sum += xe
                    startindex = endindex
                    endindex = startindex+batchsize

                #print("Epoch {0}, accuracy {1}".format(epoch,correct/total_count))
                summary = sess.run(nets['accsum'],feed_dict={nets['accuracy']:correct/total_count})
                test_writer.add_summary(summary,nets['global_step'].eval(session=sess))
                #test_writer.add_summary(summary,epoch)
                xesummary = sess.run(nets['xesum'],feed_dict={nets['xe_ave']:xe_sum/numbatches})
                test_writer.add_summary(xesummary,nets['global_step'].eval(session=sess))

            sess.run(nets['increment_current_epoch'])
            sess.run(nets['reset_current_step'])
            
        if SAVE:
            sv.saver.save(sess,'Saved_Sessions{0}/model.ckpt'.format(Saver_Folder), global_step = epoch)             
            epochsaver.save(sess,save_path = "EPOCH_SAVED{0}/model.ckpt".format(Saver_Folder), global_step = epoch)


        
def main(Learning_Rate,limit = None,music_only = False, print_extras = True, Net_or_VGG = "Net",name="",unbalanced=False, folder = "TB2", Conv1_filtersize = 9, TASK = "PREDICT_LABELS", OVERIDE_FOLDER = None,SAVE = True, padding = "SAME",poolmethod="MAXPOOL",conv1_times_hanning = False,Freeze_Model_File= None):
    #if we want to use the unbalanced set for training/eval, read that file in
    if(unbalanced):
        trainfile = "/om/user/ribeirop/audiosetDL/unbalanced_stripped/unbalanced_train_segments_Coch.hdf5"
        train_averagefile = "/om/user/ribeirop/audiosetDL/unbalanced_stripped/unbalanced_train_segments_average.npy"
    else:
        trainfile = "/om/user/ribeirop/audiosetDL/balanced_stripped/balanced_train_segments_Coch.hdf5"
        train_averagefile = "/om/user/ribeirop/audiosetDL/balanced_stripped/balanced_train_segments_average.npy"
   
    testfile =  "/om/user/ribeirop/audiosetDL/eval_stripped/eval_segments_Coch.hdf5"
    test_averagefile = "/om/user/ribeirop/audiosetDL/eval_stripped/eval_segments_average.npy"
    trainset = h5py.File(trainfile,"r")
    testset = h5py.File(testfile,"r")
    
    train_coch = trainset["/coch"] #The cochleagrams are in here
    test_coch = testset["/coch"]
    
    #    unbalanced_label_counts = np.load("/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/unbalanced_label_counts.npy")
    

    
    if (TASK == "PREDICT_LABELS"):
        multiple_labels = True
        if(music_only): #If we want to use a label set without music
            if(unbalanced):
                trainlabels_file = "/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/unbalanced_train_segments_nomusic_labels_only.hdf5"           
            else:
                trainlabels_file = "/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/balanced_train_segments_nomusic_labels_only.hdf5"
                
            testlabels_file = "/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/eval_segments_nomusic_labels_only.hdf5"

            trainlabelsSet = h5py.File(trainlabels_file,"r")
            testlabelsSet = h5py.File(testlabels_file,"r")

            trainlabels = trainlabelsSet["/labels"]
            testlabels = testlabelsSet["/labels"]  

            numlabels = 526
        else:
            trainlabels = trainset["/labels"]
            testlabels = testset["/labels"]

            numlabels = 527
    elif(TASK == "PREDICT_COUNTS"):
        multiple_labels = False
        if(unbalanced):
            trainlabels = np.load("/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/unbalanced_leaf_label_counts.npy")
        else:
            trainlabels = np.load("/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/balanced_leaf_label_counts.npy")

        testlabels = np.load("/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/eval_leaf_label_counts.npy")
        numlabels = 11
    
        
    
    COCHLEAGRAM_LENGTH = 342000#171 * 880#int(342000) # full is 342000  
    train_mean_coch = np.load(train_averagefile)[0:COCHLEAGRAM_LENGTH] #mean cochleagram
    #test_mean_coch = np.load(test_averagefile)
    
    Num_Epochs = 2000
    
    if(not limit == None):
        trainsize = limit
        testsize = limit
    else:
        trainsize = trainset.attrs.get("size")
        testsize = testset.attrs.get("size")
        limit = "full"
        
    #for unbalanced set    
    cutoff = int(trainsize*.90)
        
    b = 0
    batchsize = 100
    
    
    if Freeze_Model_File == None:
        Saved_Weights = None
        sw = "F"
    else:
        sw = "T"
        Saved_Weights = get_saved_weights(Freeze_Model_File, COCHLEAGRAM_LENGTH,numlabels,train_mean_coch,Conv1_filtersize,padding, poolmethod,conv1_times_hanning,Learning_Rate,multiple_labels)
    
    
    
    print("building graph")
    #The training of the graphs are defined here.
    with tf.Graph().as_default():
        
        if OVERIDE_FOLDER == None:
        #Get the base graph
            if (Net_or_VGG == "Net"):
                TensorBoard_Folder_train = './{4}/{0}_{6}_{1}_n{2}_lr{3}_conv1FS{5}_m{7}_hcv1{8}_sw{9}_TRAIN'.format(Net_or_VGG,name,limit,Learning_Rate,folder,Conv1_filtersize,TASK,music_only,conv1_times_hanning,sw)
                TensorBoard_Folder_test = './{4}/{0}_{6}_{1}_n{2}_lr{3}_conv1FS{5}_m{7}_hcv1{8}_sw{9}_TEST'.format(Net_or_VGG,name,limit,Learning_Rate,folder,Conv1_filtersize,TASK,music_only,conv1_times_hanning,sw)
                Saver_Folder = '/{0}_{5}_{1}_n{2}_lr{3}_conv1FS{4}_m{6}_hcv1{7}_sw{8}'.format(Net_or_VGG,name,limit,Learning_Rate,Conv1_filtersize,TASK,music_only,conv1_times_hanning,sw)
                nets = Gen_audiosetNet(COCHLEAGRAM_LENGTH,numlabels,train_mean_coch,Conv1_filtersize,padding,poolmethod,conv1_times_hanning,Saved_Weights)
            else:
                TensorBoard_Folder_train = './{4}/{0}_{4}_{1}_n{2}_lr{3}_TRAIN'.format(Net_or_VGG,name,limit,Learning_Rate,folder,TASK)
                TensorBoard_Folder_test = './{4}/{0}_{4}_{1}_n{2}_lr{3}_TEST'.format(Net_or_VGG,name,limit,Learning_Rate,folder,TASK)
                Saver_Folder = '/{0}_{4}_{1}_n{2}_lr{3}'.format(Net_or_VGG,name,limit,Learning_Rate,TASK)
                nets = Gen_VGG(COCHLEAGRAM_LENGTH,numlabels,train_mean_coch)
        else:
            TensorBoard_Folder_train = './{0}/{1}_TRAIN'.format(folder,OVERIDE_FOLDER)
            TensorBoard_Folder_test = './{0}/{1}_TEST'.format(folder,OVERIDE_FOLDER)
            Saver_Folder = '/{0}'.format(OVERIDE_FOLDER)
            if (Net_or_VGG == "Net"):
                nets = Gen_audiosetNet(COCHLEAGRAM_LENGTH,numlabels,train_mean_coch,Conv1_filtersize,padding,poolmethod,conv1_times_hanning,Saved_Weights)
            else:
                nets = Gen_VGG(COCHLEAGRAM_LENGTH,numlabels,train_mean_coch)
        
        
        #Get the loss functions
        Cross_Entropy_Train_on_Labels(nets,numlabels,Learning_Rate,multiple_labels)
    
        merged = tf.summary.merge_all()
        
        #does all the training
        if(unbalanced):
            train2(test_coch,testlabels,testsize,Saver_Folder,TensorBoard_Folder_train,TensorBoard_Folder_test,batchsize,Num_Epochs,nets,trainsize,train_coch,trainlabels,COCHLEAGRAM_LENGTH,cutoff,TASK=TASK,SAVE = SAVE)      
        else:
            train1(test_coch,testlabels,testsize,Saver_Folder,TensorBoard_Folder_train,TensorBoard_Folder_test,batchsize,Num_Epochs,nets,trainsize,train_coch,trainlabels,COCHLEAGRAM_LENGTH)
        
    print("done")
            
           
  
    
if __name__ == '__main__':
    debug = False #if true, then use the default csv file to read from. 
    if debug:
        print("running")
        learning_rate = 1e-5
        
        music_only = False 
        
        limit = 2000
        Net_or_VGG = "Net"
        #Net_or_VGG = "VGG"
        name = "t3"
        unbalanced = True
        folder = "TB2_1"
        Conv1_filtersize = 9
        TASK = "PREDICT_LABELS"
        #TASK = "PREDICT_COUNTS"
        OVERIDE_FOLDER = None#"Net_UNBALANCED91_6_nfull_lr1e-06"
        SAVE = False
        padding = "SAME"
        #padding = "VALID"
        poolmethod ="HPOOL"
        #poolmethod ="MAXPOOL"
        conv1_times_hanning = True
        
        Freeze_Model_File = "EPOCH_SAVED/Net_PREDICT_LABELS_A2UN91_5_HP9_nfull_lr1e-05_conv1FS9/model.ckpt-13"
        
        main(learning_rate,limit,music_only,Net_or_VGG=Net_or_VGG, name=name,unbalanced=unbalanced,folder=folder,Conv1_filtersize=Conv1_filtersize,TASK=TASK,OVERIDE_FOLDER=OVERIDE_FOLDER,SAVE=SAVE,padding=padding,poolmethod=poolmethod,conv1_times_hanning=conv1_times_hanning,Freeze_Model_File=Freeze_Model_File)
        
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
        
        main(learning_rate,limit,music_only,Net_or_VGG=Net_or_VGG, name=name,unbalanced=unbalanced,folder=folder,Conv1_filtersize=Conv1_filtersize,TASK=TASK,OVERIDE_FOLDER=OVERIDE_FOLDER,SAVE=SAVE,padding=padding,poolmethod=poolmethod,conv1_times_hanning=conv1_times_hanning,Freeze_Model_File)

#TODO FIX COCH SIZE EFORE RUNNING ANYTHING