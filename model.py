import torch.nn as nn 
import torch.nn.functional as F 
import torch
import torch.optim as optim
import time

class CNN(nn.Module):
    def __init__(self, conv_paras, input_dim = 28, channels = 1, classes = 10, hd = 500, bn = False):
        #conv : outputchannels, kernel_size, pooling
        super().__init__()
        self.convs = []
        self.bn = bn
        self.conv_paras = conv_paras
        self.convs.append( nn.Conv2d(channels, conv_paras[0]['oc'], conv_paras[0]['ks'], padding = conv_paras[0]['ks']//2))
        conv_output_dim = input_dim**2//conv_paras[0]['pl']**2
        for i in range(1,len(conv_paras)):
            self.convs.append( nn.Conv2d(conv_paras[i-1]['oc'], conv_paras[i]['oc'], conv_paras[i]['ks'], padding = conv_paras[i]['ks']//2) )
            conv_output_dim = conv_output_dim//conv_paras[i]['pl']**2
        conv_output_dim *= conv_paras[-1]['oc']
        if bn:
            self.bns = []
            self.bns.append( nn.BatchNorm2d(conv_paras[0]['oc']) )
            for i in range(1,len(conv_paras)):
                self.bns.append( nn.BatchNorm2d(conv_paras[i]['oc']) )
        self.conv_output_dim = conv_output_dim
        self.affine1 = nn.Linear(self.conv_output_dim, hd)
        self.affine2 = nn.Linear(hd, classes)
        
        self.Conv_layers = nn.ModuleList()
        self.Conv_layers.extend(self.convs)
        if self.bn:
            self.Conv_layers.extend(self.bns)
                
    def forward(self, x):
        bn = self.bn
        
        x = F.relu( self.convs[0](x) )
        if bn:
            x = self.bns[0](x)
        x = F.max_pool2d(x, self.conv_paras[0]['pl'])
        
        for i in range(1,len(self.convs)):
            x = F.relu( self.convs[i](x) )
            if bn:
                x = self.bns[i](x)
            x = F.max_pool2d(x, self.conv_paras[i]['pl'])
        x = F.relu(self.affine1(x.reshape(-1,self.conv_output_dim)))
        x = self.affine2(x)
        return F.log_softmax(x, dim = 1)
    
class Trainer:
    def __init__(self, model):
        self.model = model
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model = model.cuda()
            print('Cuda Enabled')
        else:
            print('Cuda is not supported. Training can be very slow.')
        self.optimizer = optim.Adam(model.parameters())
        
    def train(self, trainset, epochs, val = None, test_train_acc = True):
        model = self.model
        model.train()
        optimizer = self.optimizer
        for epoch in range(epochs):
            time_start = time.time()
            total = len(trainset)
            i = 0
            for data in trainset:  
                X, y = data 
                if self.cuda:
                    X = X.cuda()
                    y = y.cuda()
                optimizer.zero_grad() 
                output = self.model(X)
                loss = F.nll_loss(output, y)  
                loss.backward()
                optimizer.step()
                i += 1
                print('(Epcho %d / %d) Training process : %.3f%%'%(epoch+1, epochs, 100*i/total),end='\r')
            time_end = time.time()
            print('(Epcho %d / %d) Done in time %.3f s'%(epoch+1, epochs, time_end-time_start)+' '*40, end = '\r')
            msg = '(Epcho %d / %d) Done in time %.3f s  '%(epoch+1, epochs, time_end-time_start)
            if test_train_acc:
                train_acc = self.test(trainset,False)
                msg += 'Train_acc : %.3f%%  '%(train_acc*100)
            if val:
                val_acc = self.test(val, False)
                msg += 'Validation_acc : %.3f%% '%(val_acc*100)
            print(msg)
                
            
    def test(self,testset, verbose = True):
        model = self.model
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for data in testset:
                X, y = data
                if self.cuda:
                    X = X.cuda()
                    y = y.cuda()
                output = model(X)
                for idx, i in enumerate(output):
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total += 1
            if verbose:
                print("TestAccuracy: %f%%"%(100*correct/total))
        return correct/total
        
        
