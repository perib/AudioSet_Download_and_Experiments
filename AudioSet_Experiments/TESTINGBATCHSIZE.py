import tensorflow as tf
import numpy as np
import sys
import h5py
import random
import math

def main():
    #sess = tf.InteractiveSession()
    
    trainfile = "/om/user/ribeirop/audiosetDL/balanced_stripped/balanced_train_segments_Coch.hdf5"
    train_averagefile = "/om/user/ribeirop/audiosetDL/balanced_stripped/balanced_train_segments_average.npy"
    testfile = trainfile#"/om/user/ribeirop/audiosetDL/eval_stripped/eval_segments_Coch.hdf5"
    test_averagefile = train_averagefile#"/om/user/ribeirop/audiosetDL/eval_stripped/eval_segments_average.npy"
    trainset = h5py.File(trainfile)
    testset = h5py.File(testfile)
    
    #TODO: Subtract this when doing the model
    train_ave_coch = np.load(train_averagefile)
    test_ave_coch = np.load(test_averagefile)
    
    COCHLEAGRAM_LENGTH = int(342000) # full is 342000
    conv1_strides = 3
    conv2_strides = 2
    conv3_strides = 1
    conv4_strides = 1
    conv5_strides = 1
    
    trainsize = trainset.attrs.get("size")
    testsize = testset.attrs.get("size")
    
    b = 0
    batchsize =64
    
    final_filter = 512
    full_length = final_filter * math.ceil(COCHLEAGRAM_LENGTH/171/conv1_strides/conv2_strides/conv3_strides/conv4_strides/conv5_strides/4)* math.ceil(171/conv1_strides/conv2_strides/conv3_strides/conv4_strides/conv5_strides/4)
    
    nets = {}

    with tf.Graph().as_default():
        with tf.name_scope('input'):
            nets['input_to_net'] = tf.placeholder(tf.float32, [None,COCHLEAGRAM_LENGTH], name='input_to_net')
            nets['actual_labels'] = tf.placeholder(tf.float32, [None, 527], name='actual_labels')
            nets['label_indeces'] = tf.placeholder(tf.int32, [None, 15], name='label_indeces')
            
            nets['total_label_count'] = tf.placeholder(tf.int32,[1],name = 'total_label_count')
            nets['total_correct'] = tf.placeholder(tf.int32,[1],name = 'total_correct')
            
            nets['accuracy'] = tf.divide(nets['total_label_count'],nets['total_correct'])
            tf.summary.scalar("accuracy",nets['accuracy'])
            
            
            nets['reshapedCoch'] = tf.reshape(nets['input_to_net'],[-1,171,int(COCHLEAGRAM_LENGTH/171),1], name='reshape_input')

        #sess.run(tf.global_variables_initializer())
        #thing = sess.run(reshapedCoch,feed_dict={input_to_net:trainset["/coch"][0:2],actual_labels:trainset["/labels"][0:2]})



        with tf.name_scope('conv1'):
            nets['conv1'] = conv2d(nets['reshapedCoch'], Weights = weight_variable([9,9,1,96]), bias = bias_variable([96]), strides=conv1_strides, name='conv1')
            nets['maxpool1'] = maxpool2x2(nets['conv1'],k=2 , name='maxpool1')


        with tf.name_scope('conv2'):
            nets['conv2'] = conv2d(nets['maxpool1'], Weights = weight_variable([5,5,96,256]), bias = bias_variable([256]), strides=conv2_strides, name='conv2')
            nets['maxpool2'] = maxpool2x2(nets['conv2'],k=2 , name='maxpool2')

        with tf.name_scope('conv3'):
            nets['conv3'] = conv2d(nets['maxpool2'], Weights = weight_variable([3,3,256,512]), bias = bias_variable([512]), strides=conv3_strides, name='conv3')

        with tf.name_scope('conv4'):
            nets['conv4'] = conv2d(nets['conv3'], Weights = weight_variable([3,3,512,1024]), bias = bias_variable([1024]), strides=conv4_strides, name='conv4')        

        with tf.name_scope('conv5'):
            nets['conv5'] = conv2d(nets['conv4'], Weights = weight_variable([3,3,1024,final_filter]), bias = bias_variable([final_filter]),strides=conv5_strides, name='conv5')             
            nets['flattened'] = tf.reshape(nets['conv5'], [-1, full_length])

        with tf.name_scope('fc_1'):
            W_fc1 = weight_variable([full_length, 1024]) # 4,959,232
            b_fc1 = bias_variable([1024])
            nets['fc_1'] = tf.nn.relu(tf.matmul(nets['flattened'], W_fc1) + b_fc1)

            keep_prob = tf.placeholder(tf.float32)
            nets['h_fc1_drop'] = tf.nn.dropout(nets['fc_1'], keep_prob)

        with tf.name_scope('fc_2'):
            W_fc2 = weight_variable([1024, 527])
            b_fc2 = bias_variable([527])
            nets['predicted_labels'] = tf.matmul(nets['h_fc1_drop'], W_fc2) + b_fc2

        nets['cross_entroy'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels =nets['actual_labels'],logits = nets['predicted_labels'] ))
        nets['train_step'] = tf.train.AdamOptimizer(1e-4).minimize(nets['cross_entroy'])

        #num = np.sum(nets['actual_labelsi'])
        #nets['accuracy'] = tf.metrics.mean(tf.nn.in_top_k(nets['predicted_labels'],nets['actual_labelsi'],2))

        #calculating accuracys
        
        _,indeces = tf.nn.top_k(nets['predicted_labels'],15)
        nets['numCorrect'] = tf.shape(tf.sets.set_intersection(indeces,nets['label_indeces'],False).values)[0]
        
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter('./TB/train', sess.graph)
            for i in range(sys.maxsize):
                print(i)
                #if(i%100 == 0 ):
                    #print(i)                
                #get random batch
                start = random.randint(0,50)#trainsize-batchsize)
                end = start+batchsize     
                thing = sess.run(nets['train_step'],feed_dict={nets['input_to_net']:trainset["/coch"][start:end][:,0:COCHLEAGRAM_LENGTH]-train_ave_coch[0:COCHLEAGRAM_LENGTH],nets['actual_labels']:trainset["/labels"][start:end],keep_prob:.5})
                
                
                if(i%100 == 0):
                    
                    total_count = 0
                    correct = 0
                    
                    for x in range(100):
                        #get random batch
                        start = random.randint(0,50)#testsize-batchsize)
                        end = start+batchsize
                        trainBatchLabels = testset["/labels"][start:end]
    
                        indeces_list = []

                        for row in trainBatchLabels:
                            indeces= np.where(row==1)[0]
                            total_count = total_count + len(indeces)
                            indeces_list.append(np.pad(indeces,[0,15-len(indeces)],mode = 'constant',constant_values=-1))
                        
                        print(start)
                        print(end)
                        print(trainBatchLabels)
                        print(indeces_list)
                        correct = correct + sess.run(nets['numCorrect'],feed_dict={nets['input_to_net']:testset["/coch"][start:end][:,0:COCHLEAGRAM_LENGTH]-test_ave_coch[0:COCHLEAGRAM_LENGTH],nets['actual_labels']:testset["/labels"][start:end],nets['label_indeces']:indeces_list,keep_prob:1})
                    
                    #summary = sess.run(merged,feed_dict={nets['total_correct']:[correct],nets['total_label_count']:[total_count]})
                    #train_writer.add_summary(summary, i)
                    
                    print("loop {0}, accuracy {1}".format(i,correct/total_count))
                    f= open("output3.csv","a")
                    f.write("loop {0}, accuracy {1}\n".format(i,correct/total_count))
                    f.close()

            values = sess.run(nets['predicted_labels'],feed_dict={nets['input_to_net']:trainset["/coch"][b:b+1][:,0:COCHLEAGRAM_LENGTH],nets['actual_labels']:trainset["/labels"][b:b+1],keep_prob:1})
            #use eval?

        print(np.sum(trainset["/labels"][b]))
        indeces = np.argsort(values)[0][-np.sum(trainset["/labels"][b]):]
        print(indeces)
        print(np.where(trainset["/labels"][b] == True))
    
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

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
    
main()
