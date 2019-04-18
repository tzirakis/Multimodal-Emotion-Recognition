import menpo
import tensorflow as tf
import numpy as np
import os

from io import BytesIO
from pathlib import Path
from moviepy.editor import VideoFileClip
from menpo.visualize import progress_bar_str, print_progress
from moviepy.audio.AudioClip import AudioArrayClip


root_dir = Path('path_of_RECOLA')

# Inser the numbers for each video for the RECOLA Dataset
portion_to_id = dict(
    train = [], 
    valid = [],
    test  = [] 
)

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
  main(Path('path_to_save_tfrecords'))

