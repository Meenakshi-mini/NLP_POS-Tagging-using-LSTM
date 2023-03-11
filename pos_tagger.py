import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import re


def data_preprocessing(data):
  c = 2;
  X, y, X_int, y_int= [],[],[],[];
  for i in range(0,len(data)):
      if len(data[i])==1:
          continue
      data_i_split = data[i].split() 
      if data_i_split[0] == '#':
          c-=1;
      elif(c == 0):
          if i>1:
              X.append(X_int)
              y.append(y_int)
          X_int, y_int = [],[]
          X_int.append(data_i_split[1].lower())
          y_int.append(data_i_split[3])
          c = 2
      elif(c == 2):
          X_int.append(data_i_split[1].lower())
          y_int.append(data_i_split[3])
          
  X.pop(0)
  y.pop(0)
  return X,y


def prepare_sequence(seq, to_idx):    
    idxs = [to_idx[w] for w in seq]
    idxs = np.array(idxs)    
    return torch.from_numpy(idxs) 

def accuracy(test,predicted):
  total_words = 0
  correct_predictions = 0

  for i in range(0,len(test)):
    for j in range(0,len(test[i])):
      if test[i][j] == predicted[i][j]:
        correct_predictions += 1
      total_words += 1

  accuracy = correct_predictions / total_words
  return accuracy


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()        
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

        
    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_outputs = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_outputs, dim=1)        
        return tag_scores


inference = 1

f = open('/home/meenakshi/Downloads/en_atis-ud-train.conllu','r')
data = f.readlines() 
f.close() 
X,y = data_preprocessing(data)
f = open('/home/meenakshi/Downloads/en_atis-ud-test.conllu','r')
test_data = f.readlines()
f.close()
X_test,y_test = data_preprocessing(test_data)
f = open('/home/meenakshi/Downloads/en_atis-ud-dev.conllu','r')
dev_data = f.readlines()
f.close()
X_dev,y_dev = data_preprocessing(dev_data)

X_all = X + X_test + X_dev; # Considering all vocabulary

#Assigning each unique word to a unique integer
word2idx = {}
for sent in X_all:    
    for word in sent:        
        if word not in word2idx:            
            word2idx[word] = len(word2idx)

#Assigning each unique POS tag to a unique integer
postag2idx = {}
for sent in y:    
    for postag_word in sent:        
        if postag_word not in postag2idx:            
            postag2idx[postag_word] = len(postag2idx)

#Concatenating Input sentences and target POS tags
training_data = []
for i in range(0,len(X)):
    training_data.append((X[i],y[i]))        
        

#Mapping function to convert predicted POS tags index to POS tags
idx2postag = {}
for k,v in postag2idx.items():
  idx2postag[v] = k
    
 
EMBEDDING_DIM = 6
HIDDEN_DIM = 6  

if inference == 0:    
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2idx), len(postag2idx))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    
    n_epochs = 80
    for epoch in range(n_epochs):    
        epoch_loss = 0.0
        for sentence, tags in training_data:
            model.zero_grad()
            model.hidden = model.init_hidden()
    
            # Converting all sentences and target POS tags into Tensors of numerical indices
            sentence_in = prepare_sequence(sentence, word2idx)
            targets = prepare_sequence(tags, postag2idx)
    
            # forward pass to get tag scores
            tag_scores = model(sentence_in)
    
            # computing the loss and the gradients 
            loss = loss_function(tag_scores, targets)
            epoch_loss += loss.item()
            loss.backward()
            
            # update the model parameters
            optimizer.step()
            
        # printing out the average loss for every 20 epochs
        if(epoch%20 == 19):
            print("Epoch: %d, loss: %1.5f" % (epoch+1, epoch_loss/len(training_data)))    
     
    
    torch.save(model.state_dict(), "/home/meenakshi/Downloads/model_6.pth")   
    
    predicted_tags_list = []
    #Running over all test sentences
    for test_sentence in X_test:
      inputs = prepare_sequence(test_sentence, word2idx)
      tag_scores = model(inputs)  
      _, predicted_tags = torch.max(tag_scores, 1)
      predicted_tags_list.append([idx2postag[i] for i in predicted_tags.tolist()])
    print("Accuracy for test data =", accuracy(y_test,predicted_tags_list))

else:
    #Inference
    
    model1 = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2idx), len(postag2idx))
    model1.load_state_dict(torch.load('/home/meenakshi/Downloads/model_6.pth',map_location=torch.device('cpu')))
        
    input_Sentence = input('Please enter the sentence');
    cleaned_Sentence = re.sub(r'[^a-zA-Z ]','', input_Sentence).lower().split()
    inputs = prepare_sequence(cleaned_Sentence, word2idx)
    tag_scores = model1(inputs)  
    _, predicted_tags = torch.max(tag_scores, 1)
    predicted_tags_list = predicted_tags.tolist()
    for i in range(0,len(predicted_tags_list)):
        print(cleaned_Sentence[i],"\t",idx2postag[predicted_tags_list[i]])