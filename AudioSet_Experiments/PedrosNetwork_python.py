#TODO SAVE PER EPOCH IN NON UNBALANCED SET
import tensorflow as tf
import numpy as np
import h5py
import math
from math import sqrt
import argparse
import os



#Generates and returns the VGG graph (up to the predictions node)
def Gen_VGG(COCHLEAGRAM_LENGTH,numlabels,train_mean_coch):
    
    final_filter = 512
    full_length = final_filter * math.ceil(COCHLEAGRAM_LENGTH/171/2/2/2/2/2)* math.ceil(171/2/2/2/2/2)
    print(full_length)
    nets = {}
    with tf.name_scope('input'):
        nets['input_to_net'] = tf.placeholder(tf.float32, [None,COCHLEAGRAM_LENGTH], name='input_to_net')
        mean_tensor = tf.constant(train_mean_coch, dtype=tf.float32)
        nets['subtract_mean'] = tf.subtract(nets['input_to_net'],mean_tensor,name='Subtract_Mean')
        nets['reshapedCoch'] = tf.reshape(nets['subtract_mean'],[-1,171,int(COCHLEAGRAM_LENGTH/171),1], name='reshape_input')

    with tf.name_scope("accuracy"):
        nets['accuracy'] = tf.placeholder(tf.float32, (), name='acc')
        nets['accsum'] = tf.summary.scalar("accuracy",nets['accuracy'])
        
    #sess.run(tf.global_variables_initializer())
    #thing = sess.run(reshapedCoch,feed_dict={input_to_net:train_coch[0:2],actual_labels:trainlabels[0:2]})

    with tf.variable_scope('conv1') as scope:
        #Conv_1_1 and save the graphs
        tf.get_variable_scope().reuse_variables()
        nets['conv1_1_Weights'] = weight_variable([3,3,1,64])

        nets['layer1'] = variable_summaries(nets['conv1_1_Weights'])

        nets['grid'] = put_kernels_on_grid(nets['conv1_1_Weights'],8,8)
        nets['conv1_weight_image'] = tf.summary.image('conv1/kernels', nets['grid'], max_outputs=3)       
        nets['conv1_1'] = conv2d(nets['reshapedCoch'], Weights = nets['conv1_1_Weights'], bias = bias_variable([64]), strides=1, name='conv1_1')

        #conv_1_2         
        nets['conv1_2'] = conv2d(nets['conv1_1'], Weights = weight_variable([3,3,64,64]), bias = bias_variable([64]), strides=1, name='conv1_2')       
    with tf.name_scope("maxpool1"):
        nets['maxpool1'] = maxpool2x2(nets['conv1_2'],k=2 , name='maxpool1')
        
        
    with tf.name_scope('conv2'):
        nets['conv2_1'] = conv2d(nets['maxpool1'], Weights = weight_variable([3,3,64,128]), bias = bias_variable([128]), strides=1, name='conv2_1')        
        nets['conv2_2'] = conv2d(nets['conv2_1'], Weights = weight_variable([3,3,128,128]), bias = bias_variable([128]), strides=1, name='conv2_2')                  
    with tf.name_scope("maxpool2"):
        nets['maxpool2'] = maxpool2x2(nets['conv2_2'],k=2 , name='maxpool2')


    with tf.name_scope('conv3'):
        nets['conv3_1'] = conv2d(nets['maxpool2'], Weights = weight_variable([3,3,128,256]), bias = bias_variable([256]), strides=1, name='conv3_1')            
        nets['conv3_2'] = conv2d(nets['conv3_1'], Weights = weight_variable([3,3,256,256]), bias = bias_variable([256]), strides=1, name='conv3_2')       
        nets['conv3_3'] = conv2d(nets['conv3_2'], Weights = weight_variable([3,3,256,256]), bias = bias_variable([256]), strides=1, name='conv3_3')           
        nets['conv3_4'] = conv2d(nets['conv3_3'], Weights = weight_variable([3,3,256,256]), bias = bias_variable([256]), strides=1, name='conv3_4')       
    with tf.name_scope("maxpool3"):
        nets['maxpool3'] = maxpool2x2(nets['conv3_4'],k=2 , name='maxpool3')

    with tf.name_scope('conv4'):
        nets['conv4_1'] = conv2d(nets['maxpool3'], Weights = weight_variable([3,3,256,512]), bias = bias_variable([512]), strides=1, name='conv4_1')            
        nets['conv4_2'] = conv2d(nets['conv4_1'], Weights = weight_variable([3,3,512,512]), bias = bias_variable([512]), strides=1, name='conv4_2')       
        nets['conv4_3'] = conv2d(nets['conv4_2'], Weights = weight_variable([3,3,512,512]), bias = bias_variable([512]), strides=1, name='conv4_3')           
        nets['conv4_4'] = conv2d(nets['conv4_3'], Weights = weight_variable([3,3,512,512]), bias = bias_variable([512]), strides=1, name='conv4_4')       
    
    with tf.name_scope("maxpool4"):
        nets['maxpool4'] = maxpool2x2(nets['conv4_4'],k=2 , name='maxpool4')


    with tf.name_scope('conv5'):
        nets['conv5_1'] = conv2d(nets['maxpool4'], Weights = weight_variable([3,3,512,512]), bias = bias_variable([512]), strides=1, name='conv5_1')            
        nets['conv5_2'] = conv2d(nets['conv5_1'], Weights = weight_variable([3,3,512,512]), bias = bias_variable([512]), strides=1, name='conv5_2')       
        nets['conv5_3'] = conv2d(nets['conv5_2'], Weights = weight_variable([3,3,512,512]), bias = bias_variable([512]), strides=1, name='conv5_3')           
        nets['conv5_4'] = conv2d(nets['conv5_3'], Weights = weight_variable([3,3,512,512]), bias = bias_variable([512]), strides=1, name='conv5_4')       

    with tf.name_scope("maxpool5"):
        nets['maxpool5'] = maxpool2x2(nets['conv5_4'],k=2 , name='maxpool5')

    with tf.name_scope("flatten"):
        nets['flattened'] = tf.reshape(nets['maxpool5'], [-1, full_length])

        
    with tf.name_scope("keep_prob"):
        nets['keep_prob'] = tf.placeholder(tf.float32)
        
        
    with tf.name_scope('fc_1'):
        W_fc1 = weight_variable([full_length, 4096]) # 4,959,232
        b_fc1 = bias_variable([4096])
        nets['fc_1'] = tf.nn.relu(tf.matmul(nets['flattened'], W_fc1) + b_fc1)
        nets['h_fc1_drop'] = tf.nn.dropout(nets['fc_1'], nets['keep_prob'])

    with tf.name_scope('fc_2'):
        W_fc2 = weight_variable([4096, 4096]) # 4,959,232
        b_fc2 = bias_variable([4096])
        nets['fc_2'] = tf.nn.relu(tf.matmul(nets['h_fc1_drop'], W_fc2) + b_fc2)
        nets['h_fc2_drop'] = tf.nn.dropout(nets['fc_2'], nets['keep_prob'])

    with tf.name_scope('fc_3'):
        W_fc3 = weight_variable([4096, numlabels],name='W_fc3') # 4,959,232
        b_fc3 = bias_variable([numlabels],name = 'b_fc3')
        nets['predicted_labels'] = tf.nn.relu(tf.matmul(nets['h_fc2_drop'], W_fc3) + b_fc3)

    return nets

#Generates and returns our own graph (up to the predictions node)
def Gen_audiosetNet(COCHLEAGRAM_LENGTH,numlabels,train_mean_coch,Conv1_filtersize = 9):
    
    variable_list = []
    
    conv1_strides = 3
    conv2_strides = 2
    conv3_strides = 1
    conv4_strides = 1
    conv5_strides = 1
    final_filter = 512
    full_length = final_filter * math.ceil(COCHLEAGRAM_LENGTH/171/conv1_strides/conv2_strides/conv3_strides/conv4_strides/conv5_strides/4/2)* math.ceil(171/conv1_strides/conv2_strides/conv3_strides/conv4_strides/conv5_strides/4/2)
    
    nets = {}
    with tf.name_scope('input'):
        nets['input_to_net'] = tf.placeholder(tf.float32, [None,COCHLEAGRAM_LENGTH], name='input_to_net')
        mean_tensor = tf.constant(train_mean_coch, dtype=tf.float32)
        nets['subtract_mean'] = tf.subtract(nets['input_to_net'],mean_tensor,name='Subtract_Mean')
        nets['reshapedCoch'] = tf.reshape(nets['subtract_mean'],[-1,171,int(COCHLEAGRAM_LENGTH/171),1], name='reshape_input')
      
    with tf.name_scope("images"):
        nets['coch_images'] = tf.summary.image('image',nets['reshapedCoch'][0:5,:,:,:])
    
    with tf.name_scope("accuracy"):
        nets['accuracy'] = tf.placeholder(tf.float32, (), name='acc')
        nets['accsum'] = tf.summary.scalar("accuracy",nets['accuracy'])

    with tf.variable_scope('conv1') as scope:
        tf.get_variable_scope().reuse_variables()
        nets['conv1_Weights'] = weight_variable([Conv1_filtersize,Conv1_filtersize,1,96], name = 'conv1_Weights')
        nets['conv1_bias'] = bias_variable([96],name = 'bias1_Weights')  
        nets['layer1'] = variable_summaries(nets['conv1_Weights'])
        nets['grid'] = put_kernels_on_grid(nets['conv1_Weights'],16,6)
        nets['conv1_weight_image'] = tf.summary.image('conv1/kernels', nets['grid'], max_outputs=3)
        nets['conv1'] = tf.nn.local_response_normalization(conv2d(nets['reshapedCoch'], Weights = nets['conv1_Weights'], bias = nets['conv1_bias'], strides=conv1_strides, name='conv1'), depth_radius = 2)
        
    with tf.name_scope("conv1_summaries"):
        sum_list = []
        for im in range(2):
            for channel in range(2):
                sum_list.append(tf.summary.image('conv1/featuremaps_im_{0}_ch{0}'.format(im,channel), nets['conv1'][im:im+1,:,:,channel:channel+1], max_outputs=3))
                sum_list.append(tf.summary.image('conv1/weight_im{0}_ch{0}'.format(im,channel),tf.transpose(nets['conv1_Weights'],(2,0,1,3))[0:1,:,:,channel:channel+1], max_outputs=3))        
                sum_list.append(tf.summary.histogram('conv1/featuremaps_hist_im_{0}_ch{0}'.format(im,channel), nets['conv1'][im:im+1,:,:,channel:channel+1]))
        nets['conv1_sums'] = tf.summary.merge(sum_list)
            
    with tf.name_scope("maxpool1"):
        nets['maxpool1'] = maxpool2x2(nets['conv1'],k=2 , name='maxpool1')


    with tf.name_scope('conv2'):
        nets['conv2_Weights'] = weight_variable([5,5,96,256], name = 'conv2_Weights')
        nets['conv2'] = tf.nn.local_response_normalization(conv2d(nets['maxpool1'], Weights = nets['conv2_Weights'], bias = bias_variable([256],name='conv2_bias'), strides=conv2_strides, name='conv2'), depth_radius = 2)
       
    
    with tf.name_scope("maxpool2"):
        nets['maxpool2'] = maxpool2x2(nets['conv2'],k=2 , name='maxpool2')

       
    with tf.name_scope('conv3'):
        nets['conv3'] = conv2d(nets['maxpool2'], Weights = weight_variable([3,3,256,512],name = 'conv3_Weights'), bias = bias_variable([512],name='conv3_bias'), strides=conv3_strides, name='conv3')

    with tf.name_scope('conv4'):
        nets['conv4'] = conv2d(nets['conv3'], Weights = weight_variable([3,3,512,1024],name = 'conv4_Weights'), bias = bias_variable([1024],name='conv4_bias'), strides=conv4_strides, name='conv4')      

    with tf.name_scope('conv5'):
        nets['conv5'] = conv2d(nets['conv4'], Weights = weight_variable([3,3,1024,final_filter],name = 'conv5_Weights'), bias = bias_variable([final_filter],name='conv5_bias'),strides=conv5_strides, name='conv5')             

    with tf.name_scope("maxpool5"):
        nets['maxpool3'] = maxpool2x2(nets['conv5'],k=2 , name='maxpool3')
        
    with tf.name_scope("flatten"):
        nets['flattened'] = tf.reshape(nets['maxpool3'], [-1, full_length])

    with tf.name_scope('fc_1'):
        W_fc1 = weight_variable([full_length, 1024]) # 4,959,232
        b_fc1 = bias_variable([1024])
        nets['fc_1'] = tf.nn.relu(tf.matmul(nets['flattened'], W_fc1) + b_fc1)

        nets['keep_prob'] = tf.placeholder(tf.float32)
        nets['h_fc1_drop'] = tf.nn.dropout(nets['fc_1'], nets['keep_prob'])

    with tf.name_scope('fc_2'):
        W_fc2 = weight_variable([1024, numlabels])
        b_fc2 = bias_variable([numlabels])
        nets['predicted_labels'] = tf.matmul(nets['h_fc1_drop'], W_fc2) + b_fc2
    
    return nets

def Cross_Entropy_Train_on_Labels(nets,numlabels,Learning_Rate, multiple_labels = True):
    #The evaluation/training of the graph 
    with tf.variable_scope("Cross_Entropy"):
        nets['actual_labels'] = tf.placeholder(tf.float32, [None, numlabels], name='actual_labels')
        if multiple_labels:
            nets['cross_entroy'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels =nets['actual_labels'],logits = nets['predicted_labels'] ))
        else:
            nets['cross_entroy'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels =nets['actual_labels'],logits = nets['predicted_labels'] ))
            
    with tf.variable_scope("TrainStep"):
        '''
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,'''
        beta1=0.9
        beta2=0.999
        epsilon=1e-08
        nets['global_step'] = tf.Variable(0, name='global_step', trainable=False)
        nets['train_step'] = tf.train.AdamOptimizer(Learning_Rate).minimize(nets['cross_entroy'],global_step=nets['global_step'])
        #nets['train_step'] = tf.train.AdamOptimizer(learning_rate=Learning_Rate,beta1=beta1,beta2=beta2,epsilon=epsilon).minimize(nets['cross_entroy'])

        #num = np.sum(nets['actual_labelsi'])
        #nets['accuracy'] = tf.metrics.mean(tf.nn.in_top_k(nets['predicted_labels'],nets['actual_labelsi'],2))

        #calculating accuracy
    with tf.name_scope("Summaries"):
        nets['label_indeces'] = tf.placeholder(tf.int32, [None, 15], name='label_indeces')
        nets['cross_entropy_summary'] = tf.summary.scalar("cross_entropy_summary",nets['cross_entroy'])
        _,nets['indeces'] = tf.nn.top_k(nets['predicted_labels'],15)
        prediction_sums = tf.summary.histogram('predicted_labels_histogram',nets['indeces'])
        nets['numCorrect'] = tf.shape(tf.sets.set_intersection(nets['indeces'],nets['label_indeces'],False).values)[0]

        _,nets['Top_1_index'] = tf.nn.top_k(nets['predicted_labels'],1)
        nets['numCorrect_top_pred'] = tf.shape(tf.sets.set_intersection(nets['Top_1_index'],nets['label_indeces'],False).values)[0]
        
    with tf.name_scope("Count_Tracker"):
        nets['current_epoch'] = tf.Variable(0, name='current_epoch', trainable=False, dtype=tf.int32)
        nets['increment_current_epoch'] = tf.assign(nets['current_epoch'], nets['current_epoch']+1)

        nets['current_step'] = tf.Variable(0, name='current_step', trainable=False, dtype=tf.int32)
        nets['increment_current_step'] = tf.assign(nets['current_step'], nets['current_step']+1)
        nets['reset_current_step'] = tf.assign(nets['current_step'], 0)


        
        
#thanks to https://gist.githubusercontent.com/kukuruza/03731dc494603ceab0c5/raw/3d708320145df0a962cfadb95b3f716b623994e0/gist_cifar10_train.py
def put_kernels_on_grid (kernel, grid_Y, grid_X, pad = 1):

    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels])) #3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels])) #3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype = tf.uint8)

#a bunch of neat stats we can summarize
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        meansum = tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
              stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        stdv = tf.summary.scalar('stddev', stddev)
        maxsum = tf.summary.scalar('max', tf.reduce_max(var))
        minsum = tf.summary.scalar('min', tf.reduce_min(var))
        hist = tf.summary.histogram('histogram', var)
        
        return tf.summary.merge([meansum,stdv,maxsum,minsum,hist])

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
def train2(test_coch,testlabels,testsize,Saver_Folder,TensorBoard_Folder_train,TensorBoard_Folder_test,batchsize,Num_Epochs,nets,trainsize,train_coch,trainlabels,COCHLEAGRAM_LENGTH,cutoff,TASK):
    #This saver saves the graph at every epoch, and never deletes/uses them. (otherwise the superviser will delete old saves)
    epochsaver = tf.train.Saver(max_to_keep=0) #saves all the epochs
    if(not os.path.exists("EPOCH_SAVED{0}/".format(Saver_Folder))):
        try:
            os.makedirs("EPOCH_SAVED{0}/".format(Saver_Folder))
        except:
            pass
    
    #This saver saves every 5 hours but only keeps the latest 5. for backups in case we crash
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)
    sv = tf.train.Supervisor(logdir="./Saved_Sessions{0}".format(Saver_Folder),saver = saver, summary_op=None, save_model_secs = 7200,checkpoint_basename='model.ckpt')
    
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
            image_conv1 = sess.run(nets['conv1_weight_image'])
            train_writer.add_summary(image_conv1,nets['global_step'].eval(session=sess))
            #train_writer.add_summary(image_conv1,epoch)
            #print("epoch: ",epoch)
            #Train a full epoch
            
            startindex = sess.run(nets['current_step']) * batchsize #continue were we left off if we crashed
            endindex = startindex+batchsize

            #loop through the first 90% of the set
            while(endindex < cutoff):
                
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
                    _,__,correct = sess.run([nets['train_step'],nets['increment_current_step'],nets['numCorrect']],feed_dict={nets['input_to_net']:train_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:trainlabels[startindex:endindex],nets['label_indeces']:indeces_list,nets['keep_prob']:.5})
                else:
                    _,__,correct = sess.run([nets['train_step'],nets['increment_current_step'],nets['numCorrect_top_pred']],feed_dict={nets['input_to_net']:train_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:trainlabels[startindex:endindex],nets['label_indeces']:indeces_list,nets['keep_prob']:.5})
                
                #write to tensorboard
                
                summary = sess.run(nets['accsum'],feed_dict={nets['accuracy']:correct/total_count})
                #print("epoch {0}, {1}".format(epoch,correct/total_count))

                train_writer.add_summary(summary,nets['global_step'].eval(session=sess))
                
                
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
                while(endindex <= trainsize):
                    #print(startindex)
                    trainBatchLabels = trainlabels[startindex:endindex]
                    indeces_list = []
                    for row in trainBatchLabels:
                        indeces= np.where(row==1)[0]
                        total_count = total_count + len(indeces)
                        indeces_list.append(np.pad(indeces,[0,15-len(indeces)],mode = 'constant',constant_values=-1))
                    
                                        
                    if TASK == "PREDICT_LABELS":
                        addme  = sess.run(nets['numCorrect'],feed_dict={nets['input_to_net']:train_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:trainlabels[startindex:endindex],nets['label_indeces']:indeces_list,nets['keep_prob']:1})
                    else:
                        addme  = sess.run(nets['numCorrect_top_pred'],feed_dict={nets['input_to_net']:train_coch[startindex:endindex][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:trainlabels[startindex:endindex],nets['label_indeces']:indeces_list,nets['keep_prob']:1})
                   
                    correct = correct + addme    

                    startindex = endindex
                    endindex = startindex+batchsize

                #print("Epoch {0}, accuracy {1}".format(epoch,correct/total_count))
                summary = sess.run(nets['accsum'],feed_dict={nets['accuracy']:correct/total_count})
                test_writer.add_summary(summary,nets['global_step'].eval(session=sess))
                #test_writer.add_summary(summary,epoch)

            sess.run(nets['increment_current_epoch'])
            sess.run(nets['reset_current_step'])
            sv.saver.save(sess,'Saved_Sessions{0}/model.ckpt'.format(Saver_Folder), global_step = epoch)             
            epochsaver.save(sess,save_path = "EPOCH_SAVED{0}/model.ckpt".format(Saver_Folder), global_step = epoch)


        
def main(Learning_Rate,limit = None,music_only = False, print_extras = True, Net_or_VGG = "Net",name="",unbalanced=False, folder = "TB2", Conv1_filtersize = 9, TASK = "PREDICT_LABELS", OVERIDE_FOLDER = None):
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
        if(music_only): #If we want to use a label set without music
            if(unbalanced):
                trainlabels_file = "/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/balanced_train_segments_nomusic_labels_only.hdf5"
            else:
                trainlabels_file = "/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/unbalanced_train_segments_nomusic_labels_only.hdf5"

            testlabels_file = "/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/eval_segments_nomusic_labels_only.hdf5"

            trainlabelsSet = h5py.File(trainlabels_file,"r")
            testlabelsSet = h5py.File(testlabels_file,"r")

            trainlabels = trainlabelsSet["/labels"]
            testlabels = testlabelsSet["/labels"]  

            numlabels = 525
        else:
            trainlabels = trainset["/labels"]
            testlabels = testset["/labels"]

            numlabels = 527
    elif(TASK == "PREDICT_COUNTS"):
            if(unbalanced):
                trainlabels = np.load("/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/unbalanced_label_counts.npy")
            else:
                trainlabels = np.load("/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/balanced_label_counts.npy")

            testlabels = np.load("/home/ribeirop/OMFOLDER/audiosetDL/MODIFIED_LABELS/eval_label_counts.npy")
            numlabels = 15
    
        
    
    COCHLEAGRAM_LENGTH = int(342000) # full is 342000  
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
    
    print("building graph")
    #The training of the graphs are defined here.
    with tf.Graph().as_default():
        
        if OVERIDE_FOLDER == None:
        #Get the base graph
            if (Net_or_VGG == "Net"):
                TensorBoard_Folder_train = './{4}/{0}_{6}_{1}_n{2}_lr{3}_conv1FS{5}_TRAIN'.format(Net_or_VGG,name,limit,Learning_Rate,folder,Conv1_filtersize,TASK)
                TensorBoard_Folder_test = './{4}/{0}_{6}_{1}_n{2}_lr{3}_conv1FS{5}_TEST'.format(Net_or_VGG,name,limit,Learning_Rate,folder,Conv1_filtersize,TASK)
                Saver_Folder = '/{0}_{5}_{1}_n{2}_lr{3}_conv1FS{4}'.format(Net_or_VGG,name,limit,Learning_Rate,Conv1_filtersize,TASK)
                nets = Gen_audiosetNet(COCHLEAGRAM_LENGTH,numlabels,train_mean_coch,Conv1_filtersize)
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
                nets = Gen_audiosetNet(COCHLEAGRAM_LENGTH,numlabels,train_mean_coch,Conv1_filtersize)
            else:
                nets = Gen_VGG(COCHLEAGRAM_LENGTH,numlabels,train_mean_coch)
        
        
        #Get the loss functions
        Cross_Entropy_Train_on_Labels(nets,numlabels,Learning_Rate)
    
        merged = tf.summary.merge_all()
        
        #does all the training
        if(unbalanced):
            train2(test_coch,testlabels,testsize,Saver_Folder,TensorBoard_Folder_train,TensorBoard_Folder_test,batchsize,Num_Epochs,nets,trainsize,train_coch,trainlabels,COCHLEAGRAM_LENGTH,cutoff,TASK=TASK)      
        else:
            train1(test_coch,testlabels,testsize,Saver_Folder,TensorBoard_Folder_train,TensorBoard_Folder_test,batchsize,Num_Epochs,nets,trainsize,train_coch,trainlabels,COCHLEAGRAM_LENGTH)
        
    print("done")
            
           
    
def weight_variable(shape, name = None):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name)

def bias_variable(shape,name = None):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name=name)

#builds the components
def conv2d(inputtensor, Weights, bias, strides=1,name = None):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(inputtensor, Weights, strides=[1, strides, strides, 1], padding='SAME',name=name)
    x = tf.nn.bias_add(x, bias)
    return tf.nn.relu(x)

def maxpool2x2(x, k=2, name = None):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME',name=name)
    
if __name__ == '__main__':
    debug = False #if true, then use the default csv file to read from. 
    if debug:
        learning_rate = 1e-6
        music_only = False #currently also removing speech
        limit = None
        Net_or_VGG = "Net"
        name = "UNBALANCED91_6"
        unbalanced = True
        folder = "TB3"
        Conv1_filtersize = 9
        TASK = "PREDICT_LABELS"
        #TASK = "PREDICT_COUNTS"
        OVERIDE_FOLDER ="Net_UNBALANCED91_6_nfull_lr1e-06"
        main(learning_rate,limit,music_only,Net_or_VGG=Net_or_VGG, name=name,unbalanced=unbalanced,folder=folder,Conv1_filtersize=Conv1_filtersize,TASK=TASK,OVERIDE_FOLDER=OVERIDE_FOLDER)
        
    else:
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-a', '--learning_rate' , default = 1e-6)
        parser.add_argument('-b', '--limit', default = None)
        parser.add_argument('-c', '--Model', default = "Net")
        parser.add_argument('-n', '--name', default = "noname")
        parser.add_argument('-f', '--conv1filtersize', default = 9)
        parser.add_argument('-t', '--TASK',default = "PREDICT_LABELS")
        parser.add_argument('-o', '--OVERIDE_FOLDER', default = None)
        
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
        
        music_only = False
        unbalanced = True
        folder = "TB3"
                
        main(learning_rate,limit,music_only,Net_or_VGG=Net_or_VGG, name=name,unbalanced=unbalanced,folder=folder,Conv1_filtersize=Conv1_filtersize,TASK=TASK,OVERIDE_FOLDER=OVERIDE_FOLDER)
        