import time

# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-quant-shakespeare-char-minimized'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 100 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'quant-shakespeare-char-minimized-' + str(time.time())

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 12
block_size = 64 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 0.001 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model

# Quantization
weight_bit_width = 8
quant_output = False
output_bit_width = 8