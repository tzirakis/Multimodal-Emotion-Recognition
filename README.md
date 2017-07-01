# End-to-End Multimodal Emotion Recognition using Deep Neural Networks

This package provides training and evaluation code for the end-to-end multimodal emotion recognition paper. If you use this codebase in your experiments please cite:

`Tzirakis, P., Trigeorgis, G., Nicolaou, M. A., Schuller, B., & Zafeiriou, S. (2017). End-to-End Multimodal Emotion Recognition using Deep Neural Networks. arXiv preprint arXiv:1704.08619.` (https://arxiv.org/pdf/1704.08619.pdf)

## Requirements
The below requirements are needed to generate the tfrecords in the `data_generator.py` file. If you need to run the model you need only tensorflow.

  * NumPy >= 1.11.1
  * TensorFlow >= 1.1
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
  * inception_processing.py: provides methods to regularize images. 
  
The folder models contains pretrained models that were used in the paper.
