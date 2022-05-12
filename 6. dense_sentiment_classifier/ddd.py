from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# import modelcheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint
model = Sequential([
    Embedding(n_unique_words, n_dim, input_length=max_review_length),
    Flatten(),
    Dense(n_dense, activation='relu'),
    Dropout(dropout),
    Dense(n_dense, activation='relu'),
    Dropout(dropout),
    Dense(1, activation='sigmoid')
])

# fit model with modelcheckpoint and EarlyStopping
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit model with modelcheckpoint and EarlyStopping
model.fit(x_train, y_train, epochs=n_epochs, batch_size=n_batch_size, callbacks=[ modelcheckpoint, earlystopping ]) 

