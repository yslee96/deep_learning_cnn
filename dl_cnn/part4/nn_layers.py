import numpy as np
from skimage.util.shape import view_as_windows

##########
#   convolutional layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_convolutional_layer:

    def __init__(self, Wx_size, Wy_size, input_size, in_ch_size, out_ch_size, std=1e0):
    
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2),
                                  (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
        self.input_size = input_size

        self.channel_size = in_ch_size
        self.num_filters = out_ch_size

        self.filter_width = Wx_size
        self.filter_height = Wy_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        filter_width = self.filter_width
        filter_height = self.filter_height
        num_filters = self.num_filters

        batch_num = x.shape[0]
        out_width = x.shape[2] - filter_width + 1
        out_height = x.shape[3] - filter_height +1

        out = np.zeros((batch_num, num_filters, out_width, out_width))

        for batch in range(batch_num):
            for filters in range(num_filters):
                image = x[batch]
                filt = self.W[filters]
                y = view_as_windows(image, filt.shape)
                y = y.reshape((out_width, out_height, -1))
                conv_result = y.dot(filt.reshape(-1,1)) + self.b[0,filters,0,0]
                conv_result = np.squeeze(conv_result, axis=2)
                
                out[batch, filters] += conv_result  

        return out      

    def backprop(self, x, dLdy):

        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
  
        filter_width = self.filter_width
        filter_height = self.filter_height
        num_filters = self.num_filters

        batch_num = x.shape[0]
        channel_size = x.shape[1]
        out_width = x.shape[2] - filter_width +1
        out_height =x.shape[3] - filter_height+1

        #dLdx
        
        dLdx = np.zeros(x.shape)
        for batch in range(batch_num):
            for filters in range(num_filters):
                for channel in range(channel_size):

                    dy = dLdy[batch, filters]
                    dy = np.pad(dy, ((filter_height - 1, filter_width - 1), (filter_width - 1, filter_height - 1)), 'constant', constant_values=0)
                    

                    filt = self.W[filters, channel]
                    filt = filt[::-1][:,::-1]

                    w = dy.shape[0] - filt.shape[0] + 1
                    h = dy.shape[1] - filt.shape[0] + 1

                    y = view_as_windows(dy, filt.shape)
                    y = y.reshape(w,h,-1)

                    bp_result = y.dot(filt.reshape(-1,1))
                    bp_reuslt = np.squeeze(bp_result, axis=2)
                    
                    dLdx[batch,channel] += bp_reuslt
        
        #dLdW
        
        dLdW = np.zeros(self.W.shape)
        for batch in range(batch_num):
            for filters in range(num_filters):
                for channel in range(channel_size):

                    image = x[batch, channel]
                    filt = dLdy[batch, filters]

                    w = image.shape[0] - filt.shape[0] + 1
                    h = image.shape[1] - filt.shape[1] + 1

                    y = view_as_windows(image, (out_width, out_height))
                    y = y.reshape((w,h,-1))
                    bp_result = y.dot(filt.reshape(-1,1))
                    bp_result = np.squeeze(bp_result, axis=2)

                    dLdW[filters, channel] += bp_result

        #dLdb

        dLdb = np.zeros(self.b.shape)
        for filters in range(num_filters):
            dLdb[0,filters, 0, 0] += np.sum(dLdy[:,filters,:,:])

        return dLdx, dLdW, dLdb   

##########
#   max pooling layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        batch_num = x.shape[0]
        channel_size = x.shape[1]
        out_width  = int((x.shape[2]-self.pool_size) / self.stride + 1)
        out_height  = int((x.shape[3]-self.pool_size) / self.stride + 1)  

        out = np.zeros((batch_num, channel_size, out_width, out_height))

        for batch in range(batch_num):
            for channel in range(channel_size):
                image = x[batch, channel]
                y = view_as_windows(image, (2,2), step=2)

                for w in range(out_width):
                    for h in range(out_height):
                        out[batch, channel, w, h] += np.max(y[w,h])

        return out

    def backprop(self, x, dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        batch_num = x.shape[0]
        channel_size = x.shape[1]
        out_width  = int((x.shape[2]-self.pool_size) / self.stride + 1)
        out_height  = int((x.shape[3]-self.pool_size) / self.stride + 1)  

        dLdy = dLdy.reshape((batch_num, 28, 13, 13))
        dLdx = np.zeros((batch_num, channel_size, x.shape[2], x.shape[3]))

        for batch in range(batch_num):
            for channel in range(channel_size):
                image = x[batch, channel]
                y = view_as_windows(image, (2,2), step=2)

                for w in range(out_width):
                    for h in range(out_height):
                        tmp = np.argmax(y[w, h])
                        if(tmp == 0):
                            arg = (0,0)
                        elif (tmp == 1):
                            arg = (0,1)
                        elif (tmp == 2):
                            arg = (1,0)
                        else:
                            arg = (1,1)

                        dLdx[batch, channel, arg[0] +2*w, arg[1]+2*h] += dLdy[batch, channel, w, h]

        return dLdx



##########
#   fully connected layer
##########
# fully connected linear layer.
# parameters: weight matrix matrix W and bias b
# forward computation of y=Wx+b
# for (input_size)-dimensional input vector, outputs (output_size)-dimensional vector
# x can come in batches, so the shape of y is (batch_size, output_size)
# W has shape (output_size, input_size), and b has shape (output_size,)

class nn_fc_layer:

    def __init__(self, input_size, output_size, std=1):
        # Xavier/He init
        self.W = np.random.normal(0, std/np.sqrt(input_size/2), (output_size, input_size))
        self.b=0.01+np.zeros((output_size))

    def forward(self,x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        x = x.reshape((x.shape[0],-1))
        out = x@self.W.T + self.b

        return out

    def backprop(self,x,dLdy):

        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        x = x.reshape((x.shape[0],-1))

        dLdx = np.dot(dLdy, self.W)
        dLdW = np.dot(dLdy.T, x)
        dLdb = np.sum(dLdy, axis=0, keepdims=True)
        dLdb = dLdb[0]

        return dLdx,dLdW,dLdb

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

##########
#   activation layer
##########
#   This is ReLU activation layer.
##########

class nn_activation_layer:
    
    # performs ReLU activation
    def __init__(self):
        pass
    
    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        out = np.maximum(0, x)
        
        return out
    
    def backprop(self, x, dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        dLdx = dLdy
        dLdx[x<=0] = 0
        
        return dLdx


##########
#   softmax layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_softmax_layer:

    def __init__(self):
        pass

    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        n = x.shape[0]
        m = x.shape[1]
        out = np.zeros((n,m))
        x_exp = np.exp(x)

        for i in range(n):
            tmp = np.sum(x_exp[i])
            for j in range(m):
                out[i,j] = x_exp[i,j] / tmp
        
        return out


    def backprop(self, x, dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        n = x.shape[0]
        m = x.shape[1]
        y = np.zeros((n,m))
        x_exp = np.exp(x)

        for i in range(n):
            tmp = np.sum(x_exp[i])
            for j in range(m):
                y[i,j] = x_exp[i,j] / tmp


        n = y.shape[0]
        dLdx = y
        for i in range(n):
            dydx = y[i] * np.eye(y[i].size) - y[i].reshape(x.shape[1],1) @ y[i].reshape(1,x.shape[1])
            dLdx[i]= dLdy[i] @ dydx 


        return dLdx
   

##########
#   cross entropy layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_cross_entropy_layer:

    def __init__(self):
        pass

    def forward(self, x, y):

        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        n = x.shape[0]
        losses = np.zeros((n,))
        for i in range(n):
            losses[i] = -np.log(x[i,y[i]])
        
        out = np.mean(losses)

        return out

    def backprop(self, x, y):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        n = y.shape[0]
        m = x.shape[1]
        dLdx = np.zeros((n,m)) 
        for i in range(n):
            dLdx[i,y[i]] -= 1
        
        dLdx /= n

        return dLdx
