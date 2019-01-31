
from keras.preprocessing import sequence
import keras.backend as K
from keras import layers, models
from run_models import utils
import sys

# Get data and additional inputs
n_epochs = utils.get_arg('n_epochs')
X_train, X_test, y_train, y_test = utils.get_data(sys.argv[1])

def lstm_model(maxlen=40, num_classes=3, embedding_size=50, vocab_size=15342, n_neurons=128):
     # Extract dims from data
     #maxlen = X_train.shape[1]
     #num_classes = y_train.shape[1] # negative, neutral, positive 
     # Shape does NOT include batch size, just num features (which is equal to maxlen/num_timesteps here)
     # Should be of shape (?, 10):  "A shape tuple (integer), not including the batch size."
     sequences = layers.Input(shape=(maxlen,),dtype='int32')
     # Embeddings: Input_dim=vocab_size and output_dim=embedding_size
     # In this case, vocab_size = Glove_vocab_size
     embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_size, embeddings_initializer='uniform', trainable=True)
     # INITIALIZE LSTM 1: Set return sequences to true
     lstm1 = layers.LSTM(units=n_neurons, activation='tanh', return_sequences=True, trainable=True, go_backwards=False)
     # INITIALIZELSTM 2
     lstm2 = layers.LSTM(units=n_neurons, activation='tanh', return_sequences=False, trainable=True, go_backwards=False)
     # INITIALISE final desnse layer
     dense = layers.Dense(units=num_classes, activation='softmax')
     # BUILD MODEL
     op1 = embedding(inputs=sequences)
     op2 = lstm1(inputs=op1)
     op3 = lstm2(inputs=op2)
     op4 = dense(inputs=op3)
     return models.Model(inputs=sequences, outputs=op4)
      
if __name__ == '__main__':
     model = lstm_model()
     model.summary()
     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
     model.fit(x=X_train, y=y_train, epochs=n_epochs, batch_size=128, shuffle=True)


# #results = lstm_model.fit(x=X_train, y=y_train, epochs=n_epochs, batch_size=256, shuffle=True, verbose=2)