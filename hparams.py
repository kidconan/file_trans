from text import symbols

# Text
text_cleaners = ['english_cleaners']

# Mel
n_mel_channels = 80
num_mels = 80

# FastSpeech
vocab_size = 1024
N = 6
Head = 2
d_model = 512
# d_model = 384
duration_predictor_filter_size = 256
duration_predictor_kernel_size = 3
dropout = 0.1

word_vec_dim = 512
# word_vec_dim = 384
# word_vec_dim = 384
encoder_n_layer = 6
encoder_head = 2
encoder_conv1d_filter_size = 1536
max_sep_len = 2048
encoder_output_size = 512
# encoder_output_size = 384
decoder_n_layer = 6
decoder_head = 2
decoder_conv1d_filter_size = 1536
decoder_output_size = 512
# decoder_output_size = 384
fft_conv1d_kernel = 3
fft_conv1d_padding = 1
duration_predictor_filter_size = 256
duration_predictor_kernel_size = 3
dropout = 0.1

# Train
alignment_path = "./BN_alignments"
checkpoint_path = "./BN_model_new"
logger_path = "./BN_logger"
mel_ground_truth = "./BN_mels"
data_path = './data/data/BZNSYP'

# alignment_path = "./alignments"
# checkpoint_path = "./model_new"
# logger_path = "./logger"
# mel_ground_truth = "./mels"
# data_path = './data/data/LJSpeech-1.1'

batch_size = 16
epochs = 50
n_warm_up_step = 4000

learning_rate = 1e-3
weight_decay = 1e-6
grad_clip_thresh = 1.0
decay_step = [500000, 1000000, 2000000]

save_step = 10
log_step = 5
clear_Time = 20


