import torch.nn as nn
import torch.nn.functional as F
import torch

class intra_proposal(nn.Module):
    def __init__(self,d,h,w):
        nn.Module.__init__(self)
        self.d = d
        self.h = h
        self.w = w

        self.batchnorm = nn.BatchNorm2d(self.d)
        self.relu = nn.ReLU()

        self.position_conv2d_f = nn.Conv2d(d,d,[1,1])
        self.position_conv2d_g = nn.Conv2d(d,d,[1,1])
        self.position_conv2d_last = nn.Conv2d(d,d,[1,1])        
        
        self.channel_conv2d_f = nn.Conv2d(d,d,[1,1])
        self.channel_conv2d_g = nn.Conv2d(d,d,[1,1])
        self.channel_conv2d_last = nn.Conv2d(d,d,[1,1])

        torch.nn.init.kaiming_normal_(self.position_conv2d_f.weight.data)
        torch.nn.init.kaiming_normal_(self.position_conv2d_g.weight.data)
        torch.nn.init.kaiming_normal_(self.position_conv2d_last.weight.data)
        torch.nn.init.kaiming_normal_(self.position_conv2d_f.weight.data)
        torch.nn.init.kaiming_normal_(self.position_conv2d_g.weight.data)
        torch.nn.init.kaiming_normal_(self.position_conv2d_last.weight.data)    
        

    def forward(self, input):
        """
        input N*D*H*W
        """
        N = input.shape[0]
        p_h1 = self.position_conv2d_f(input)
        p_h1 = self.batchnorm(p_h1)

        p_h1 = p_h1.view(N,self.d,-1)
        position_attention = torch.bmm(torch.transpose(p_h1,1,2),p_h1)
        
        p_g = torch.transpose(self.batchnorm(self.position_conv2d_g(input) \
            ).view(N,self.d,-1),1,2)

        p_h2 = torch.bmm(position_attention,p_g)        
        p_h2 = torch.transpose(p_h2,1,2).view(N,self.d,self.h,self.w)
        p_h2 = self.position_conv2d_last(p_h2)

        p_h2 = self.batchnorm(p_h2)
        p_h2 = self.relu(p_h2)

        c_h1 = self.channel_conv2d_f(input)
        c_h1 = self.batchnorm(c_h1)
        c_h1 = c_h1.view(N,self.d,-1)

        channel_attention = torch.bmm(c_h1,torch.transpose(c_h1,1,2))
        
        c_g = self.batchnorm(self.channel_conv2d_g(input) \
            ).view(N,self.d,-1)
        
        c_h2 = torch.bmm(channel_attention,c_g)
        c_h2 = c_h2.view(N,self.d,self.h,self.w)
        c_h2 = self.channel_conv2d_last(c_h2)
        c_h2 = self.batchnorm(c_h2)
        c_h2 = self.relu(c_h2)

        return input + p_h2 + c_h2


class inter_proposal(nn.Module):
    def __init__(self,d,h,w):
        nn.Module.__init__(self)
        self.d = d
        self.h = h
        self.w = w

        self.position_conv2d_f = nn.Conv2d(d,d,[1,1])
        self.position_conv2d_g = nn.Conv2d(d,d,[1,1])
        self.position_pool2d =  nn.MaxPool2d(self.h,self.w)
        self.position_conv2d_last = nn.Conv2d(d,d,[1,1])        
        
        self.channel_conv2d_f = nn.Conv2d(d,d,[1,1])
        self.channel_conv2d_f_d1 = nn.Conv2d(1,1,[1,1])

        self.channel_conv2d_g = nn.Conv2d(1,1,[1,1])
        self.channel_pool1d =  nn.MaxPool1d(self.d)
        self.channel_conv2d_last = nn.Conv2d(d,d,[1,1])

    def forward(self, input):
        """
        input N*D*H*W
        """
        N = input.shape[0]
        p_theta = self.position_conv2d_f(input)
        p_theta = torch.transpose(p_theta,0,1).contiguous()
        p_theta = p_theta.view(self.d,N*self.h*self.w)
        p_theta = torch.transpose(p_theta,0,1)

        pooled_input = self.position_pool2d(input).view(N,self.d,1,1)
        p_phi = self.position_conv2d_f(pooled_input)
        p_phi = p_phi.view(N,-1)
        p_phi = torch.transpose(p_phi,0,1)

        p_g = self.position_conv2d_g(pooled_input)
        p_g = p_g.view(N,-1)

        p = torch.mm(torch.mm(p_theta,p_phi),p_g).view(N,self.h*self.w,self.d)
        p = torch.transpose(p,1,2).contiguous().view(N,self.d,self.h,self.w)


        c_theta = self.position_conv2d_f(input)
        c_theta = c_theta.view(N*self.d,self.h*self.w)

        pooled_input = torch.transpose(input.view(N,self.d,-1),1,2)
        pooled_input = self.channel_pool1d(pooled_input).view(N,1,self.h,self.w)

        c_phi = self.channel_conv2d_f_d1(pooled_input)
        c_phi = c_phi.view(N,self.h*self.w)
        c_phi = torch.transpose(c_phi,0,1)

        c_g = self.channel_conv2d_g(pooled_input)
        c_g = c_g.view(N,self.h*self.w)

        c = torch.mm(torch.mm(c_theta,c_phi),c_g).view(N,self.d,self.h,self.w)


        return input + p + c


if __name__ == '__main__':
    print('Test MagNet')
    size = [20,3,16,16]
    data = torch.randn(size)

    model = intra_proposal(3,16,16)
    assert model(data).shape == tuple(size),'wrong size' 
    print model(data).shape

    model = inter_proposal(3,16,16)
    assert model(data).shape == tuple(size),'wrong size' 
    
    print model(data).shape
