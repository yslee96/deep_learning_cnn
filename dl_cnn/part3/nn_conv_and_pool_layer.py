import numpy as np
from skimage.util.shape import view_as_windows


#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######

class nn_convolutional_layer:

    def __init__(self, filter_width, filter_height, input_size, in_ch_size, num_filters, std=1e0):
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * filter_width * filter_height / 2),
                                  (num_filters, in_ch_size, filter_width, filter_height))
        self.b = 0.01 + np.zeros((1, num_filters, 1, 1))
        self.input_size = input_size

        #######
        ## If necessary, you can define additional class variables here
        #######

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    #######
    # Q1. Complete this method
    #######
    def forward(self, x):
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

    #######
    # Q2. Complete this method
    #######
    def backprop(self, x, dLdy):
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

    #######
    ## If necessary, you can define additional class methods here
    #######


class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        #######
        ## If necessary, you can define additional class variables here
        #######

    #######
    # Q3. Complete this method
    #######
    def forward(self, x):
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

    #######
    # Q4. Complete this method
    #######
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

    #######
    ## If necessary, you can define additional class methods here
    #######


# testing the implementation

# data sizes
batch_size = 8
input_size = 32
filter_width = 3
filter_height = filter_width
in_ch_size = 3
num_filters = 8

std = 1e0
dt = 1e-3

# number of test loops
num_test = 20

# error parameters
err_dLdb = 0
err_dLdx = 0
err_dLdW = 0
err_dLdx_pool = 0

for i in range(num_test):
    # create convolutional layer object
    cnv = nn_convolutional_layer(filter_width, filter_height, input_size, in_ch_size, num_filters, std)

    x = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    # dLdx test
    print('dLdx test')
    y1 = cnv.forward(x)
    y2 = cnv.forward(x + delta)

    bp, _, _ = cnv.backprop(x, np.ones(y1.shape))

    exact_dx = np.sum(y2 - y1) / dt
    apprx_dx = np.sum(delta * bp) / dt
    print('exact change', exact_dx)
    print('apprx change', apprx_dx)

    err_dLdx += abs((apprx_dx - exact_dx) / exact_dx) / num_test * 100

    # dLdW test
    print('dLdW test')
    W, b = cnv.get_weights()
    dW = np.random.normal(0, 1, W.shape) * dt
    db = np.zeros(b.shape)

    z1 = cnv.forward(x)
    _, bpw, _ = cnv.backprop(x, np.ones(z1.shape))
    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_dW = np.sum(z2 - z1) / dt
    apprx_dW = np.sum(dW * bpw) / dt
    print('exact change', exact_dW)
    print('apprx change', apprx_dW)

    err_dLdW += abs((apprx_dW - exact_dW) / exact_dW) / num_test * 100

    # dLdb test
    print('dLdb test')

    W, b = cnv.get_weights()

    dW = np.zeros(W.shape)
    db = np.random.normal(0, 1, b.shape) * dt

    z1 = cnv.forward(x)

    V = np.random.normal(0, 1, z1.shape)

    _, _, bpb = cnv.backprop(x, V)

    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_db = np.sum(V * (z2 - z1) / dt)
    apprx_db = np.sum(db * bpb) / dt

    print('exact change', exact_db)
    print('apprx change', apprx_db)
    err_dLdb += abs((apprx_db - exact_db) / exact_db) / num_test * 100

    # max pooling test
    # parameters for max pooling
    stride = 2
    pool_size = 2

    mpl = nn_max_pooling_layer(stride=stride, pool_size=pool_size)

    x = np.arange(batch_size * in_ch_size * input_size * input_size).reshape(
        (batch_size, in_ch_size, input_size, input_size)) + 1
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    print('dLdx test for pooling')
    y1 = mpl.forward(x)
    dLdy = np.random.normal(0, 10, y1.shape)
    bpm = mpl.backprop(x, dLdy)

    y2 = mpl.forward(x + delta)

    exact_dx_pool = np.sum(dLdy * (y2 - y1)) / dt
    apprx_dx_pool = np.sum(delta * bpm) / dt
    print('exact change', exact_dx_pool)
    print('apprx change', apprx_dx_pool)

    err_dLdx_pool += abs((apprx_dx_pool - exact_dx_pool) / exact_dx_pool) / num_test * 100

# reporting accuracy results.
print('accuracy results')
print('conv layer dLdx', 100 - err_dLdx, '%')
print('conv layer dLdW', 100 - err_dLdW, '%')
print('conv layer dLdb', 100 - err_dLdb, '%')
print('maxpool layer dLdx', 100 - err_dLdx_pool, '%')