import torch.nn as nn 
import torch.nn.functional as F 
import torch
import torch.optim as optim

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
                
    def forward(self, x):
        bn = self.bn
        
        x = F.relu( self.convs[0](x) )
        if bn:
            x = self.bns[0](x)
        x = F.max_pool2d(x, self.conv_paras[0]['pl'])
        
        for i in range(1,len(self.convs)):
            x = F.relu( self.convs[i](x) )
            if bn:
                x = self.bns[i][x]
            x = F.max_pool2d(x, self.conv_paras[i]['pl'])
        x = F.relu(self.affine1(x.reshape(-1,self.conv_output_dim)))
        x = self.affine2(x)
        return F.log_softmax(x, dim = 1)
    
class Trainer:
    def __init__(self, model, cuda = False):
        self.model = model
        self.cuda = cuda
        if cuda:
            self.model = self.model.cuda()
        
        self.optimizer = optim.Adam(model.parameters())
        
    def train(self, trainset, epochs, val = None):
        model = self.model
        model.train()
        optimizer = self.optimizer
        for epoch in range(epochs):
            
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
                print('Epcho %d Training process : %.3f%%'%(epoch+1, i/total),end='\r')
            print('Epcho %d Done'%(epoch+1))
            
    def test(testset):
        model = self.model
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for data in testset:
                X, y = data
                output = model(X)
                for idx, i in enumerate(output):
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total += 1

            print("TestAccuracy: %f"%(correct/total))
        return correct/total
        
        
