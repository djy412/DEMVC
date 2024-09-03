# Parameter configurations
BATCH_SIZE = 40
EPOCHS = 5 #200
CLASSIFIER_EPOCHS = 5
MODEL_FILENAME1 = './data/Encoder.pt'
MODEL_FILENAME2 = './data/DecoderC.pt'
MODEL_FILENAME3 = './data/DecoderP.pt'
EARLY_STOP_THRESH = 50 # default 30
# Learning rate for both encoder and decoder
LR = 0.003443387869251986
LAMBDA = 0.01
LATENT_DIM = 10
NUM_CLASSES = 10
# Load data in parallel by choosing the best num of workers for your system
WORKERS = 8
dataset_name = 'Dataset: Multi-Market'