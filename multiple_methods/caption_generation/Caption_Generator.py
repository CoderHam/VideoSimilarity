
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import json
import os

from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, GRU, Dropout
from keras.optimizers import RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.models import Model
from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from Attention import Attention_Layer
from Multimodel_layer import Multimodel_Layer


# In[2]:

class Caption_Generator:
    
    def __init__(self):
        self.captions = []
        self.captions_in_each_video = []
        self.word2id = {}
        self.id2word = {}
        self.max_sentence_length = 0
        self.vocabulary_size = 0
        self.batch_size = 10
        self.embedding_output_shape = 256
        
    ################################################################################################
    def read_data(self, n_batch):
        print("loading Data for new Batch... ")
        files = [] 
    
        #reading captions
        with open('data/training_label.json') as data_file:
            training_labels = json.load(data_file)
        
        
        self.captions_in_each_video = []
        for i in n_batch:
            files.append(training_labels[i]['id'])
            for j in range(len(training_labels[i]['caption'])):
                training_labels[i]['caption'][j] = "<s> "+training_labels[i]['caption'][j]+" <e>" 
                self.captions.append(training_labels[i]['caption'][j].lower().split(' '))
            self.captions_in_each_video.append(len(training_labels[i]['caption']))

        
        #reading video features
        video_features = np.zeros((len(files),80,4096))
        
        video_features[0] = np.load("data/training_data/feat/"+files[0]+".npy")

        for i in range(1,len(files)):
            video_features[i] = np.load("data/training_data/feat/"+files[i]+".npy")
        
        print("Data Loaded Successfully.....")

        return video_features
    ################################################################################################
    def create_vocabulary(self):

        print("creating vocabulary...")
        labels = []
        with open('datatraining_label.json') as data_file:
            training_labels = json.load(data_file)
        
        for i in range(len(training_labels)):
            for j in range(len(training_labels[i]['caption'])):
                training_labels[i]['caption'][j] = "<s> "+training_labels[i]['caption'][j]+" <e>" 
                labels.append(training_labels[i]['caption'][j].lower().split(' '))
        
        self.max_sentence_length = 1 + max([len(caption) for caption in labels])
        print("\t Max sentence length : ", self.max_sentence_length)
         
        #computing char2id and id2char vocabulary
        index = 0
        for caption in labels:
            for word in caption:
                if word not in self.word2id:
                    self.word2id[word] = index
                    self.id2word[index] = word
                    index += 1
                    
        
        self.vocabulary_size = len(self.word2id)
    
       
            
    ################################################################################################
    def transform_inputs(self, video_features):
        #transforming the no of samples of video features equal to no of samples of captions
        new_features = np.zeros((len(self.captions), 80, 4096))
        for i in range(len(self.captions_in_each_video)):
            for j in range(self.captions_in_each_video[i]):
                new_features[j] = video_features[i]
                
        return new_features
            
    
    ################################################################################################
    def one_of_N_encoding(self): 
        print("encoding inputs...")      
        #creating caption tensor that is a matrix of size numCaptions x maximumSentenceLength x wordVocabularySize
        encoded_tensor = np.zeros((len(self.captions), self.max_sentence_length, self.vocabulary_size), dtype=np.float16)
        label_tensor = np.zeros((len(self.captions), self.max_sentence_length, self.vocabulary_size), dtype =np.float16)
        #one-hot-encoding
        for i in range(len(self.captions)):
            for j in range(len(self.captions[i])):
                encoded_tensor[i, j, self.word2id[self.captions[i][j]]] = 1
                if j<len(self.captions[i])-1:
                    label_tensor[i,j,self.word2id[self.captions[i][j+1]]] = 1
                
        return encoded_tensor, label_tensor
    
    ################################################################################################
    def embedding_layer(self, input_data):
        print("embedding inputs....")
        model = Sequential()
        model.add(Dense(self.embedding_output_shape, input_shape = (self.max_sentence_length, self.vocabulary_size)))
        model.add(Activation('relu'))
        model.compile('rmsprop','mse')
        embedding_weights = model.get_weights()
        output_array = model.predict(input_data)
        self.embedding_weights = model.get_weights()
        output_weights = np.asarray(self.embedding_weights[0]).T
        self.embedding_weights[0] = output_weights
        self.embedding_weights[1] = np.ones((self.vocabulary_size,))
        return output_array
    
    ################################################################################################
    def data_preprocessing(self, n_batch):
        #########################Preprocessing Data##############################
        #print("Data Preprocessing.......")
        #print("\tReading data.......")
        video_features = self.read_data(n_batch)
        video_features = self.transform_inputs(video_features)
        #print("\tvideo features : ",video_features.shape)
        #print("\tCaptions : ", len(self.captions))
        #print("\tCreating Vocabulary......")
        #self.create_vocabulary()

        # one-hot encoding of captions
        #print("\tEncoding Captions......")
        encoded_tensor, label_tensor = self.one_of_N_encoding()
        #print("\tEncoded Captions : ",encoded_tensor.shape)

        # embedding the one-hot encoding of each word into 512
        #print("\tEmbedding Captions.......")
        embedded_input = self.embedding_layer(encoded_tensor)

        #print("\tEmbedding Weights : ", np.asarray(self.embedding_weights[0]).shape)

        #print("\tEmbedded_captions : ",embedded_input.shape)
        
        return video_features, embedded_input, label_tensor
        
    ################################################################################################    
    def build_model(self, video_features, embedded_input):
        #########################training model##################################
        print('Building Sentence Generator Model...')

        input1 = Input(shape=(embedded_input.shape[1],embedded_input.shape[2]), dtype='float32')
        #input2 = Input(shape=(visual_features.shape[0],visual_features.shape[1]), dtype='float32')
        input2 = Input(shape=(video_features.shape[1], video_features.shape[2]), dtype='float32')
        
        model = Sequential()
        
        layer1 = GRU(256, return_sequences = True, input_shape = (embedded_input.shape[1],embedded_input.shape[2]), activation = 'relu')(input1)
        
        attention_layer = Attention_Layer(output_dim = 32)([layer1, input2])

        multimodel_layer = Multimodel_Layer(output_dim = 1024)([layer1,attention_layer])

        dropout = Dropout(0.5)(multimodel_layer)

        layer2 = TimeDistributed(Dense(activation = 'tanh', units = 256))(dropout)

        softmax_layer = Dense(units = self.vocabulary_size, activation = 'softmax', weights = self.embedding_weights)(layer2)
        
        model = Model(inputs = [input1, input2], outputs = [softmax_layer])
        
        '''
        # We also specify here the optimization we will use, in this case we use RMSprop with learning rate 0.001.
        # RMSprop is commonly used for RNNs instead of regular SGD.
        # categorical_crossentropy is the same loss used for classification problems using softmax. (nn.ClassNLLCriterion)
        '''
        model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(lr=0.001))

        print(model.summary()) # Convenient function to see details about the network model.

        return model
    
    ################################################################################################    
    def train(self):       
        
        batches = np.arange(1450)
        #########################training model##################################
        for epoch in range(10):
            print("\n\n\nEpoch : ",epoch+1)
            np.random.shuffle(batches)
            batch = 0
            for iteration in range(int(1450/self.batch_size)):
                if batch+self.batch_size >= 1450:
                    n_batch = batches[batch:-1]
                else:    
                    n_batch = batches[batch:(batch+self.batch_size)]
                batch += self.batch_size
                self.captions = []
                video_features, embedded_input, label_tensor = self.data_preprocessing(n_batch)
                if(iteration == 0 and epoch == 0):
                    model = caption_generator.build_model(video_features, embedded_input)
                # define the checkpoint
                filepath="Sentence_Generator_Model_Results/word-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
                checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
                callbacks_list = [checkpoint]

                print("\n\n###########Training the model on epoch : ", epoch+1, " batch : ", iteration+1 ," ###########\n\n")
                model.fit(x = [embedded_input,video_features], y = label_tensor, batch_size = 64, epochs= 5, callbacks = callbacks_list)
            self.save_model(model,epoch)
            
            
        return model
    
    ################################################################################################    
    def save_model(self, model, epoch):
        # serialize model to JSON
        filename = "Sentence_Generator_Model_Results/model_epoch_"+str(epoch)+".h5"
        #with open("batch_model.json", "w") as json_file:
            #json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(filename)
        print("Saved model to disk")
        
    ################################################################################################    
    def load_model(self, model, epoch):
        # load weights into new model
        filename = "Sentence_Generator_Model_Results/model_epoch_"+str(epoch)+".h5"
        model.load_weights(filename)
        print("Loaded model from disk")
        return model
    
    ################################################################################################    
    def test(self, model, epoch):

        print("word : ",self.id2word[0])
        test_captions = []
        with open('datatesting_public_label.json') as data_file:
            testing_labels = json.load(data_file)
        
        files = []
        self.captions_in_each_video = []

        for i in range(len(testing_labels)):
            files.append(testing_labels[i]['id'])
            for j in range(len(testing_labels[i]['caption'])):
                test_captions.append(testing_labels[i]['caption'][j].lower().split(' '))
            self.captions_in_each_video.append(j)
        
        encoded_tensor = np.zeros((len(test_captions), self.max_sentence_length, self.vocabulary_size), dtype=np.float16)
        encoded_tensor[:,0,0] = 1

        print("number of files : ",len(files))
        #reading video features
        video_features = np.zeros((len(files),80,4096))
        
        print("shape : ",np.load("data/testing_data/feat/"+files[0]+".npy").shape)

        for i in range(len(files)):
            video_features[i] = np.load("data/testing_data/feat/"+files[i]+".npy")

        new_features = np.zeros((len(self.captions), 80, 4096))
        for i in range(len(self.captions_in_each_video)):
            for j in range(self.captions_in_each_video[i]):
                new_features[j] = video_features[i]
        
        new_features = np.reshape(new_features, (len(self.captions)*80, 1, 4096))
        

        #print("new_features : ", new_features.shape)
        encoded_tensor = np.repeat(encoded_tensor, 80, axis=0)

        embedded_input = self.embedding_layer(encoded_tensor)

        print("embedded_input : ", embedded_input.shape)
        print("video_features : ", new_features.shape)

        model  = self.build_model()
        model = self.load_model(model)

        output = model.predict([embedded_input[:200,:,:], new_features[:200,:,:]])
        
        with open("Model_Results/Results/generated_text_epoch"+str(epoch)+".txt", "a") as fileHandler:
            
            for i in range(200):
                text = ""
                for j in range(41):
                    word = np.argmax(output[i,j,:])
                    text += self.id2word[word]
                    text += " "
                fileHandler.write("Generated text for example ",i," : ", text)
                fileHandler.write("\n")
            fileHandler.close()    
            

    ################################################################################################



# In[3]:

caption_generator = Caption_Generator()


# In[ ]:

caption_generator.create_vocabulary()


# In[ ]:

model = caption_generator.train()


# In[ ]:



