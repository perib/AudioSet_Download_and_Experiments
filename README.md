# MIT MSRP2017 AudioSet Project
This project was completed during the MIT MSRP 2017 summer internship program. I worked with graduate student Jenelle Feather in Josh McDermott's Laboratory for Computational Audition.

### AudiosetDL
Tools for downloading and preprocessing audio from Youtube.

### Audioset Experiments
Tools for training a deep convolutional neural network on the dataset.


## Abstract

### Sound Classification with Convolutional Neural Networks
Pedro Ribeiro, Jenelle Feather and Josh McDermott

In recent years, convolutional neural networks (CNNs) have reached human-like performance on
real world tasks. When trained on image, speech, and music classification tasks, CNNs achieve
state of the art performance and, in some cases, replicate properties of biological sensory
systems. Currently, most labeled audio databases are restricted in either the scope of labels (i.e
containing only speech or music) or in size. Thus, networks are typically trained only on speech
or music tasks, limiting the extent to which they can be compared to the entire auditory system.
Here, we use Google AudioSet, a newly released collection of multiclass labeled sound files
taken from Youtube, to train a convolutional neural network on a broad audio classification task.
AudioSet contains over 2 million audio samples, each sample classified with up to 15 labels out
of 527. Our network consists of 5 hierarchical convolutional layers with local response
normalization followed by pooling after the first, second, and fifth layers. The model was
implemented in Tensorflow and optimized via the Adam optimizer with a cross entropy loss
function. We explored changes to the optimization and architecture such as varying the learning
rate, filter sizes, and pooling type. Notably, changing max pooling to a weighted average
pooling with a hanning window did not decrease performance on the task, and led to an increase
in performance for some learning rates. Confusion patterns revealed implicit knowledge of sound
category structure (for instance, the trained networks confused genres of music). Future work
will further explore architecture and hyperparameter optimization and training on new tasks,
such as predicting number of labels, using the same dataset and architecture. Additionally, we
will compare performance and classification errors to human behavior on a similar task and
synthesize sounds from the hidden layers.
