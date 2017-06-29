import os

# Pamateres Initialization
init_lr = 0.0001
batch_size = 1
seq_length = 150
hidden_units = 256
model = 'video'
#model = 'audio'
#model = 'both'
#preprocess = 'nothing'
#preprocess = 'distorted_crop110x110_dist_color2_vars'
#preprocess = 'newest_models_augment_27-03-2017'
#preprocess = 'nothing_from_scratch'
#preprocess = 'NEW_GOLD_distorted_crop_' + str(seq_length)
preprocess = 'nothing_compare_' + str(seq_length)

#portion = 'test'
#portion = 'valid'
portion = 'train'

folder = '/vol/atlas/homes/pt511/ckpt/multimodal_emotion_recognition/' + model + '/' + \
         model + '_h=' + str(init_lr) + '_b=' + str(batch_size) +'_s=' + str(seq_length) + \
         'hu=' + str(hidden_units) + '_' + preprocess

train_dir = folder + '/train'
if portion == 'valid':
  eval_dir = folder + '/eval'
else:
  eval_dir = folder + '/eval_train'

#eval_dir = folder + '/eval_test'
eval_dir = folder + '/eval_train_valid_new'

print(train_dir)
print(eval_dir)

# Start training
os.system("python emotion_eval.py " +
    "--log_dir=" + eval_dir + "  "
    "--hidden_units=" + str(hidden_units) + "  "
    "--seq_length=" + str(seq_length) + " "
    "--checkpoint_dir=" + train_dir + " "
    "--portion=" + portion + " "
    "--model=" + model)
