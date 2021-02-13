from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau
import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import h5py

from utils.utils import prepare_output_dirs, print_config, write_config
from utils.feature_extraction import read_data
from utils.model import create_model
from utils.visdata import save_history, plot_confusion_matrix
from utils.SGDW import SGDW

from config import parse_opts

config = parse_opts()
config = prepare_output_dirs(config)

print_config(config)
write_config(config, os.path.join(config.save_dir, 'config.json'))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= config.device

####################################################################
####################################################################

# Parameters
song_samples = 660000
genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4, 
          'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}

num_genres = len(genres)
# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath=os.path.join(config.checkpoint_dir,'{epoch:03d}-{val_loss:.2f}.hdf5'),
    verbose=1,
    save_best_only=True)

# Helper: Save results.
csv_logger = CSVLogger(os.path.join(config.log_dir,'training.log'))

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=config.early_stopping_patience)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir=config.log_dir)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

####################################################################
####################################################################

def main():
    # Read the data
    X, y = read_data(config.dataset_path, genres, song_samples)

    # Transform to a 3-channel image
    X_stack = np.squeeze(np.stack((X,) * 3, -1))
    X_train, X_test, y_train, y_test = train_test_split(X_stack, y, test_size=0.3, random_state=42, stratify = y)
    
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Histogram for train and test 
    values, count = np.unique(np.argmax(y_train, axis=1), return_counts=True)
    plt.bar(values, count)

    values, count = np.unique(np.argmax(y_test, axis=1), return_counts=True)
    plt.bar(values, count)
    plt.savefig( os.path.join(config.save_dir,'histogram.png'),format='png', bbox_inches='tight')

    # Training step
    input_shape = X_train[0].shape
    
    model = create_model(input_shape, num_genres)
    model.summary()

    optimizer = SGDW(momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])

    hist = model.fit(X_train, y_train, batch_size = config.batch_size, epochs = config.num_epochs, validation_data = (X_test, y_test), 
        callbacks=[checkpointer, early_stopper, tensorboard, csv_logger, reduce_lr])

    # Evaluate
    score = model.evaluate(X_test, y_test, verbose = 0)
    # Plot graphs
    save_history(hist, os.path.join(config.save_dir, 'evaluate.png') )

    # Save the confusion Matrix
    preds = np.argmax(model.predict(X_test), axis = 1)
    y_orig = np.argmax(y_test, axis = 1)
    conf_matrix = confusion_matrix(preds, y_orig)

    keys = OrderedDict(sorted(genres.items(), key=lambda t: t[1])).keys()
    plot_confusion_matrix( os.path.join(config.save_dir,'cm.png'), conf_matrix, keys, normalize=True)

if __name__ == '__main__':
    main()
