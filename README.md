# End-to-End Multimodal Emotion Recognition using Deep Neural Networks

This package provides training and evaluation code for the end-to-end multimodal emotion recognition paper. If you use this codebase in your experiments please cite:

`P. Tzirakis, G. Trigeorgis, M. A. Nicolaou, B. Schuller and S. Zafeiriou, "End-to-End Multimodal Emotion Recognition using Deep Neural Networks," in IEEE Journal of Selected Topics in Signal Processing, vol. PP, no. 99, pp. 1-1.` (http://ieeexplore.ieee.org/document/8070966/)


## Requirements
Below are listed the required modules to run the code.

  * Python <= 2.7
  * NumPy >= 1.11.1
  * TensorFlow <= 0.12
  * Menpo >= 0.6.2
  * MoviePy >= 0.2.2.11
 
## Content
This repository contains the files:
  * model.py: contains the audio and video networks.
  * emotion_train.py: is in charge of training.
  * emotion_eval.py: is in charge of evaluating.
  * data_provider.py: provides the data.
  * data_generator.py: creates the tfrecords from '.wav' files
  * metrics.py: contains the concordance metric used for evaluation.
  * losses.py: contains the loss function of the training.
  * inception_processing.py: provides functions for visual regularization. 
  
The multimodal model can be downloaded from here : https://www.doc.ic.ac.uk/~pt511/emotion_recognition_model.zip 
