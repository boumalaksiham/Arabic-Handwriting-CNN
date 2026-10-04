"""Portable CNN retraining: select with training-only validation, report test once."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent

def load_split(directory, images, labels):
    pixels = np.loadtxt(directory / images, delimiter=',', dtype=np.float32)
    target = np.loadtxt(directory / labels, delimiter=',', dtype=np.float64).reshape(-1)
    if pixels.ndim != 2 or pixels.shape != (len(target), 1024):
        raise ValueError('Expected one 1024-pixel row per label')
    if not np.isfinite(pixels).all() or np.any((pixels < 0) | (pixels > 255)):
        raise ValueError('Pixels must be finite values in [0, 255]')
    if not np.isfinite(target).all() or np.any(target != np.floor(target)) or np.any((target < 1) | (target > 28)):
        raise ValueError('Labels must be integers from 1 to 28')
    return pixels.reshape(-1, 32, 32, 1) / 255.0, target.astype(np.int32) - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'my_dataset')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'Models/retrained')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    import tensorflow as tf
    from tensorflow.keras import Sequential, layers, callbacks
    tf.keras.utils.set_random_seed(args.seed)
    names = ['csvTrainImages 13440x1024.csv', 'csvTrainLabel 13440x1.csv',
             'csvTestImages 3360x1024.csv', 'csvTestLabel 3360x1.csv']
    x, y = load_split(args.data_dir, *names[:2])
    train_rows, validation_rows = train_test_split(np.arange(len(y)), test_size=.2,
        stratify=y, random_state=args.seed)
    model = Sequential([
        layers.Input(shape=(32, 32, 1)),
        layers.Conv2D(32, 3, padding='same', activation='relu'), layers.BatchNormalization(),
        layers.MaxPooling2D(2), layers.Dropout(.3),
        layers.Conv2D(64, 3, activation='relu'), layers.BatchNormalization(),
        layers.MaxPooling2D(2), layers.Dropout(.3),
        layers.Conv2D(128, 3, activation='relu'), layers.BatchNormalization(),
        layers.MaxPooling2D(2), layers.Dropout(.4), layers.Flatten(),
        layers.Dense(256, activation='relu'), layers.BatchNormalization(),
        layers.Dropout(.5), layers.Dense(28, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = model.fit(x[train_rows], y[train_rows], batch_size=32, epochs=args.epochs,
        validation_data=(x[validation_rows], y[validation_rows]),
        callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                   callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.2, patience=2, min_lr=1e-6)])
    model.save(args.output_dir / 'model4arabic.keras')
    # This script never passes the supplied test rows to fit or callbacks.
    x_test, y_test = load_split(args.data_dir, *names[2:])
    predicted = model.predict(x_test, verbose=0).argmax(axis=1)
    report = {'seed':args.seed, 'tensorflow_version':tf.__version__,
        'evaluation_role':'supplied_test_split_after_training_only_validation_selection',
        'caveat':'This supplied test split was examined in the historical notebook; it is not a fresh external cohort.',
        'training_row_indices':train_rows.tolist(), 'validation_row_indices':validation_rows.tolist(),
        'test_count':len(y_test), 'test_accuracy':float(np.mean(predicted == y_test)),
        'classification_report':classification_report(y_test, predicted, labels=list(range(28)), output_dict=True, zero_division=0),
        'confusion_matrix':confusion_matrix(y_test, predicted, labels=list(range(28))).tolist(),
        'test_labels':y_test.tolist(), 'test_predictions':predicted.tolist(),
        'history':{k:[float(v) for v in values] for k,values in history.history.items()},
        'input_sha256':{n:hashlib.sha256((args.data_dir/n).read_bytes()).hexdigest() for n in names}}
    (args.output_dir / 'evaluation.json').write_text(json.dumps(report, indent=2))
    print('Saved checkpoint and evaluation report in', args.output_dir)

if __name__ == '__main__':
    main()
