#!/usr/bin/env python3
"""Train a TensorFlow model on the merged CSP dataset produced by the CSP solver.

Expects an NPZ file with keys: csp_states (N,9,9), csp_policy (N,9,9,9), csp_value (N,)

Example:
  python3 train_csp_tf.py snapshots --epochs 10 --batch 32
"""
import argparse
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models


def load_dataset(npz_path):
    d = np.load(npz_path)
    X = d['csp_states']  # (N,9,9)
    Y_policy = d['csp_policy']  # (N,9,9,9)
    Y_value = d['csp_value']  # (N,)
    return X.astype(np.int8), Y_policy.astype(np.float32), Y_value.astype(np.float32)


def preprocess(X):
    # convert to one-hot channels: 9 channels for values 1..9 plus mask channel for emptiness
    N = X.shape[0]
    X_chan = np.zeros((N, 9, 9, 10), dtype=np.float32)
    for i in range(1, 10):
        X_chan[:, :, :, i-1] = (X == i).astype(np.float32)
    # mask channel: 1 if cell is empty (candidate cell), else 0
    X_chan[:, :, :, 9] = (X == 0).astype(np.float32)
    return X_chan


def build_model(input_shape=(9,9,10)):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, padding='same', activation='swish')(inp)
    x = layers.Conv2D(64, 3, padding='same', activation='swish')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='swish')(x)
    # policy head: per-cell 9-way softmax
    p = layers.Conv2D(128, 1, activation='swish', padding='same')(x)
    p = layers.Conv2D(9, 1, activation=None, padding='same')(p)
    p = layers.Softmax(axis=-1, name='policy')(p)

    # value head: per-grid scalar representing solvability probability
    v = layers.GlobalAveragePooling2D()(x)
    v = layers.Dense(64, activation='swish')(v)
    v = layers.Dense(1, activation='sigmoid', name='value')(v)

    model = models.Model(inputs=inp, outputs=[p, v])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
        loss_weights={'policy': 0.1, 'value': 0.5},
        metrics={'policy': 'accuracy', 'value': 'accuracy'}
    )
    return model

def build_model2(input_shape=(9,9,10)):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    # Policy head: per-cell 9-way softmax
    p = layers.Conv2D(128, 1, padding='same')(x)
    p = layers.BatchNormalization()(p)
    p = layers.Activation('swish')(p)
    p = layers.Conv2D(9, 1, padding='same')(p)
    p = layers.Softmax(axis=-1, name='policy')(p)

    # Value head: per-grid scalar representing solvability probability
    v = layers.GlobalAveragePooling2D()(x)
    v = layers.Dense(64)(v)
    v = layers.BatchNormalization()(v)
    v = layers.Activation('swish')(v)
    v = layers.Dense(1, activation='sigmoid', name='value')(v)

    model = models.Model(inputs=inp, outputs=[p, v])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
        loss_weights={'policy': 0.1, 'value': 0.5},
        metrics={'policy': 'accuracy', 'value': 'accuracy'}
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npz', help='Path to merged CSP dataset NPZ')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--out', default='snapshots/strategy3/csp_tf_model.keras')
    args = parser.parse_args()

    model = build_model2()
    model.summary()
    
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch}")
        for item in os.listdir(args.npz):
            full_path = os.path.join(args.npz, item)
            
            X_raw, Y_policy, Y_value = load_dataset(full_path)
            print('Loaded', X_raw.shape[0], 'examples')
            if X_raw.shape[0] == 0:
                raise SystemExit('No examples found in NPZ')
        
            X = preprocess(X_raw)
            # targets: policy is (N,9,9,9), model policy output shape is (N,9,9,9)
            # ensure policy sums are valid (add small smoothing to prevent NaNs)
            Y_policy = Y_policy.astype(np.float32)
            # value is scalar 0/1
            Y_value = Y_value.reshape(-1, 1).astype(np.float32)
        
            # split train/val
            N = X.shape[0]
            idx = np.arange(N)
            np.random.shuffle(idx)
            split = int(N * 0.8)
            train_idx = idx[:split]
            val_idx = idx[split:]
        
            history = model.fit(
                X[train_idx], {'policy': Y_policy[train_idx], 'value': Y_value[train_idx]},
                validation_data=(X[val_idx], {'policy': Y_policy[val_idx], 'value': Y_value[val_idx]}),
                epochs=1,
                batch_size=args.batch
            )

    # save model
    model.save(args.out)
    print('Saved model to', args.out)


if __name__ == '__main__':
    main()
