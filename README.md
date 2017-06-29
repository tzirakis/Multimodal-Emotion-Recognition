# Multimodal-Emotion-Recognition

This package provides training and evaluation code for the end-to-end multimodal emotion recognition paper. If you use this codebase in your experiments please cite:

`@article{tzirakis2017end,
  title={End-to-End Multimodal Emotion Recognition using Deep Neural Networks},
  author={Tzirakis, Panagiotis and Trigeorgis, George and Nicolaou, Mihalis A and Schuller, Bj{\"o}rn and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:1704.08619},
  year={2017}
}`

## Requirements
The below requirements are needed to generate the wav files in the `data_generator.py` file. If you need to run the model you need only tensorflow.

  * NumPy >= 1.11.1
  * TensorFlow >= 1.1
  * Menpo >= 0.6.2
  * MoviePy >= 0.2.2.11
 
## Content
  * model.py: contains the audio and video networks.
  * emotion_train.py: is in charge of training.
  * emotion_eval.py: is in charge of evaluating.
  * data_provider.py: provides the data.
  * data_generator.py: creates the tfrecords from '.wav' files
  * metrics.py: contains the concordance metric used for evaluation.
  * losses.py: contains the loss function of the training.
