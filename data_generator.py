import menpo
import tensorflow as tf
import numpy as np
import menpo.io as mio
import os
import math

from io import BytesIO
from pathlib import Path
from moviepy.editor import VideoFileClip
from menpo.visualize import progress_bar_str, print_progress
from menpo.image import Image
from menpo.shape import PointCloud
from sklearn import preprocessing
from moviepy.audio.AudioClip import AudioArrayClip
from sklearn.externals import joblib

# Used to find landmarks
from menpodetect.dlib import load_dlib_frontal_face_detector
from menpodetect.ffld2 import load_ffld2_frontal_face_detector
from menpodetect.dlib.conversion import pointgraph_to_rect
from dlib import shape_predictor
from menpo.shape import PointCloud
from os.path import isdir, join

root_dir = Path('/vol/atlas/homes/gt108/db/RECOLA_CNN')

portion_to_id = dict(
    train = [25, 15, 16 ,17 ,18 ,21 ,23 ,37 ,39 ,41 ,46 ,50 ,51 ,55 ,56, 60], # 25
    valid = [14, 19, 24, 26, 28, 30, 34 ,40, 42, 43, 44, 45, 52, 64, 65],
    test  = [54, 53, 13, 20, 22, 32, 38, 47, 48, 49, 57, 58, 59, 62, 63] # 54, 53
)

# scaler = joblib.load('training_standartization.pkl')

def get_samples(subject_id):
    arousal_label_path = root_dir / 'Ratings_affective_behaviour_CCC_centred/arousal/{}.csv'.format(subject_id)
    valence_label_path = root_dir / 'Ratings_affective_behaviour_CCC_centred/valence/{}.csv'.format(subject_id)

    clip = VideoFileClip(str(root_dir / "Video_recordings_MP4/{}.mp4".format(subject_id)))

    subsampled_audio = clip.audio.set_fps(16000)

    audio_frames = []
    for i in range(1, 7501):
        time = 0.04 * i

        audio = np.array(list(subsampled_audio.subclip(time - 0.04, time).iter_frames()))
        audio = audio.mean(1)[:640]

        audio_frames.append(audio.astype(np.float32))

    arousal = np.loadtxt(str(arousal_label_path), delimiter=',')[:, 1][1:]
    valence = np.loadtxt(str(valence_label_path), delimiter=',')[:, 1][1:]

    return audio_frames, np.dstack([arousal, valence])[0].astype(np.float32)

def get_jpg_string(im):
    # Gets the serialized jpg from a menpo `Image`.
    fp = BytesIO()
    menpo.io.export_image(im, fp, extension='jpg')
    fp.seek(0)
    return fp.read()

def _int_feauture(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feauture(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_sample(writer, subject_id):
    subject_name = 'P{}'.format(subject_id)
#    os.system('mkdir {}{}'.format(p_out, subject_name))

    for i, (audio, label) in enumerate(zip(*get_samples(subject_name))):

        example = tf.train.Example(features=tf.train.Features(feature={
                    'sample_id': _int_feauture(i),
                    'subject_id': _int_feauture(subject_id),
                    'label': _bytes_feauture(label.tobytes()),
                    'raw_audio': _bytes_feauture(audio.tobytes()),
                }))

        writer.write(example.SerializeToString())
        del audio, label

def main(directory):
  for portion in portion_to_id.keys():
    print(portion)

    for subj_id in print_progress(portion_to_id[portion]):

      writer = tf.python_io.TFRecordWriter(
          (directory / 'tf_records' / portion / '{}.tfrecords'.format(subj_id)
          ).as_posix())
      serialize_sample(writer, subj_id)

if __name__ == "__main__":
  main(Path('/vol/atlas/homes/pt511/db/RECOLA_audio_only'))

