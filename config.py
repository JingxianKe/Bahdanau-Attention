# define the data file name
DATA_FNAME = "fra.txt"
# define the batch size
BATCH_SIZE = 512
# define the vocab size for the source and the target
# text vectorization layers
SOURCE_VOCAB_SIZE = 15_000
TARGET_VOCAB_SIZE = 15_000
# define the encoder configurations
ENCODER_EMBEDDING_DIM = 512
ENCODER_UNITS = 512
# define the attention configuration
ATTENTION_UNITS = 512
# define the decoder configurations
DECODER_EMBEDDING_DIM = 512
DECODER_UNITS = 1024
# define the training configurations
EPOCHS = 100
LR_START = 1e-4
LR_MAX = 1e-3
WARMUP_PERCENT = 0.15
# define the patience for early stopping
PATIENCE = 10
# define the output path
OUTPUT_PATH = "output"







