import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import torch.nn as nn
import torch
from torch.autograd import Variable
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix
from sklearn.preprocessing import OneHotEncoder

INPUT_SIZE = 1#3
SEQ_SIZE = 30
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 3
learning_rate = 0.001
num_epochs = 50
Data_type = 'Daily' ##'Daily','Monthly', 'Quarterly'

################# Data Preparation ###########################
# Test Sine wave


time = np.arange(0, 100, 0.1);
amplitude  = np.sin(time)
#plt.plot(time, amplitude)
#plt.title('Sine wave')
X = amplitude
#Y = X[SEQ_SIZE:]
#Y = np.concatenate((Y, [X[:SEQ_SIZE]))
df = pd.DataFrame(X, columns = ['X'])
df['Y'] = df['X'].shift(-SEQ_SIZE)
df['Direction_'] = ((df['Y'].shift(-1)-df['Y'])/df['Y']).shift(1)
df['Direction_'] = df['Direction_'].fillna(0)
df['Direction_'][0] = 0

boundary = 0.01
df['Direction'] = df['Direction_']
df['Direction'][df['Direction_']>boundary] = 2
df['Direction'][df['Direction_']<-boundary] = 0
df['Direction'][(df['Direction_']<=boundary) & (df['Direction_']>= -boundary) ] = 1

############################################### One-hot Encoder
enc = OneHotEncoder(handle_unknown='ignore')

enc.fit(np.array(df['Direction']).reshape(-1,1))

label=enc.transform(np.array(df['Direction']).reshape(-1,1)).toarray()
df['Direction1'] = label[:,0]
df['Direction2'] = label[:,1]
df['Direction3'] = label[:,2]
df = df.dropna()

###############################################



Previous_X = []
for i in range(SEQ_SIZE, len(df)):
    Previous_X.append(df['X'][i-SEQ_SIZE:i])
    

Previous_X = np.array(Previous_X)
target_Y = np.array(df['Y'][:-SEQ_SIZE]).reshape(-1,1)
Y = []

for i in range (len(df)-SEQ_SIZE):
    Y.append([df['Direction1'][i],df['Direction2'][i],df['Direction3'][i]])
    
Y = np.array(Y)

target_Y.shape
Y.shape
Previous_X.shape
Y_sum = np.concatenate([target_Y,Y],1)

############################################### Train-test split

train_X, test_X, train_y, test_y = train_test_split(Previous_X,
                                                    Y_sum, test_size=0.2, shuffle=False)

Y_sum[:,1:4]

################################################

#Y = Y.reshape(-1,SEQ_SIZE,INPUT_SIZE)
"""
Y_rms = RunningMeanStd()
X_rms = RunningMeanStd()
X_rms.update(X_sum)
Y_rms.update(Y)
Y_norm_train = (Y - Y_rms.mean)/np.sqrt(Y_rms.var)
X_norm_train = (X_sum - X_rms.mean)/np.sqrt(X_rms.var)
X_train = X_norm_train[:1955]
Y_train = Y_norm_train[:1955]

X_test = X_norm_train[1955:]
Y_test = Y[1955:]
X_test.shape
Y_test.shape

X_train, y_train = np.array(X_train), np.array(Y_train)
"""

class RNN(nn.Module):
    def __init__(self, i_size, h_size, n_layers, o_size):
        super(RNN, self).__init__()

        self.rnn = nn.GRU(
            input_size=i_size,
            hidden_size=h_size,
            num_layers=n_layers
        )
        self.out = nn.Linear(h_size, o_size)
        self.softmax = nn.Softmax(dim=1)
        self.regression = nn.Linear(h_size, 1)

    def forward(self, x, h_state):
        r_out, hidden_state = self.rnn(x, h_state)
        
        #print("hidden_state", hidden_state)
        hidden_size = hidden_state[-1].size(-1)
        #print("r_out=",r_out.view(-1, hidden_size).shape)
        #print("hidden_size",hidden_size)
        r_out = r_out.view(-1,30,hidden_size)
        #print("r_out=",r_out.shape)
        outs = self.out(r_out)
        outs = self.softmax(outs)
        outs2 = self.regression(r_out)

        return outs, outs2, hidden_state

rnn = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

optimiser = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.BCELoss()#nn.MSELoss()
criterion2 = nn.MSELoss()
hidden_state = None

#################################################################################################


mini_batch_size = 128
number_of_batch = len(train_X)//mini_batch_size +1
#################################################################################################

for epoch in range(num_epochs):
    for j in range (number_of_batch):
        train_X_all = np.array_split(train_X,number_of_batch)
        train_y_all = np.array_split(train_y,number_of_batch)
        
        inputs = Variable(torch.from_numpy(train_X_all[j]).float().view(-1,SEQ_SIZE,INPUT_SIZE))
        inputs.shape
        #print("X_train shape =",torch.from_numpy(X_train).float().view(-1,SEQ_SIZE,INPUT_SIZE))
        labels = Variable(torch.from_numpy(train_y_all[j][:,1:4]).float())
        labels.shape
        #labels = labels[30:]
        train_y_magnitude = Variable(torch.from_numpy(train_y_all[j][:,0]).float()).view(-1,1)
        #print("shape =",inputs.shape)
        output, output2, hidden_state = rnn(inputs, hidden_state)
        output.shape
        output = output.view(-1,SEQ_SIZE,OUTPUT_SIZE)
        output = output[:,SEQ_SIZE-1]
        output2 = output2.view(-1,SEQ_SIZE,1)
        output2 = output2[:,SEQ_SIZE-1]
        hidden_state[0].detach
        hidden_state[1].detach
        #output.view(-1).shape
        labels.shape
        output.shape
        output2.shape
        loss = criterion(output, labels.detach()) + criterion2(output2, train_y_magnitude.detach())
        optimiser.zero_grad()
        loss.backward(retain_graph=True)                     # back propagation
        optimiser.step()                                     # update the parameters
    
    print('epoch {}, loss {}'.format(epoch,loss.item()))

######################################################
##################################################################

inputs = Variable(torch.from_numpy(test_X).float().view(-1,SEQ_SIZE,INPUT_SIZE))

outs, outs2, b = rnn(inputs, hidden_state)
outs = outs[:,SEQ_SIZE-1]
outs2 = outs2[:,SEQ_SIZE-1]
#predicted_stock_price = np.reshape(predicted_stock_price.detach().numpy(), (test_inputs.shape[0], 1))
outs.shape
#predict_out = predict_out.view(120,3)
_, predict_y = torch.max(outs, 1)

test_y = enc.inverse_transform(test_y[:,1:4])
test_y = test_y.reshape(-1).astype(int)

print ('prediction accuracy', accuracy_score(test_y, predict_y.data.numpy()))
print ('macro precision', precision_score(test_y.data, predict_y.data, average='macro'))
print ('micro precision', precision_score(test_y.data, predict_y.data, average='micro'))
print ('macro recall', recall_score(test_y.data, predict_y.data, average='macro'))
print ('micro recall', recall_score(test_y.data, predict_y.data, average='micro'))

outs2 = outs2.data.numpy()
outs2 = outs2.reshape(-1)
#outall = np.concatenate([train_y[:,0],outs2],0)
plt.figure(1, figsize=(12, 5))
plt.plot(time[:-2*SEQ_SIZE-1][-len(test_y):],outs2, color = 'red', label = 'Pred')
plt.plot(time[:len(train_y)],train_y[:,0], color = 'blue', label = 'Real')
#plt.plot(outall, color = 'green', label = 'Real')

"""
outs_.shape
outs_ = outs_.reshape(-1,SEQ_SIZE,1)
outs_ = outs_[:,SEQ_SIZE-1]
outs_ = outs_.detach().numpy()

loss2 = criterion(torch.from_numpy(outs_), torch.from_numpy(Y_test))
print("Testing Lost = ", loss2.item())


check_accuracy = np.array(check_accuracy)
        
accuracy = check_accuracy[SEQ_SIZE:].sum()/len(outs_[SEQ_SIZE:])

print("accuracy = ", accuracy)
print("Direction_accuracy = ", Direction_accuracy)

"""
