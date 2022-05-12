
# create model with embedding layer, flatten layer, 2 dense layer and classification layer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Dropout

model = Sequential([
    Embedding(n_unique_words, n_dim, input_length=max_review_length),
    Flatten(),
    Dense(n_dense, activation='relu'),
    Dropout(dropout),
    Dense(n_dense, activation='relu'),
    Dropout(dropout),
    Dense(1, activation='sigmoid')
])

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

