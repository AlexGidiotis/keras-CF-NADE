from keras.engine import Layer, InputSpec
from keras import backend as K
import tensorflow as tf
from keras import initializers
from keras import regularizers
from keras import constraints


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    

class NADE(Layer):
    """
    """

    def __init__(self,
        hidden_dim,
        activation,
        weight_decay=0.0,
        kernel_regularizer=regularizers.l2(0.0),
        bias=False, **kwargs):


        self.init = initializers.get('uniform')

    
        self.bias = bias
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.weight_decay = weight_decay
        super(NADE, self).__init__(**kwargs)


    def build(self, input_shape):
        self.input_dim1 = input_shape[1]
        self.input_dim2 = input_shape[2]

        self.W = self.add_weight(shape=(self.input_dim1,self.input_dim2,self.hidden_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name))
        if self.bias:
            self.c = self.add_weight(shape=(self.hidden_dim,),
                                     initializer=self.init,
                                     name='{}_c'.format(self.name))

        self.Q = self.add_weight(shape=(self.input_dim1,self.hidden_dim),
                                 initializer=self.init,
                                 name='{}_Q'.format(self.name))

        if self.bias:
            self.b = self.add_weight(shape=(self.input_dim1,self.input_dim2),
                                     initializer=self.init,
                                     name='{}_b'.format(self.name))


        self.V = self.add_weight(shape=(self.hidden_dim,self.input_dim1,self.input_dim2),
                                 initializer=self.init,
                                 name='{}_V'.format(self.name))

        super(NADE, self).build(input_shape)


    def call(self, x):

        x = K.cumsum(x[:, :, ::-1], axis=2)[:, :, ::-1]
        # x.shape = (?,6040,5)
        # W.shape = (6040, 5, 500)
        # c.shape = (500,)
        if self.bias:
            output_ = tf.tensordot(x, self.W, axes=[[1, 2], [0, 1]]) + self.c
        else:
            output_ = tf.tensordot(x, self.W, axes=[[1, 2], [0, 1]])
        output_ = tf.reshape(output_, [-1,self.hidden_dim])
        #tf.cast(indices, tf.float32)
        # output_.shape = (?,500)

        input_mask = K.sum(x, axis = 2)
        # input_mask.shape = (?,6040)
        # Q.shape = (6040, 500)
        output_masked = K.dot(input_mask, self.Q) 
        # output_masked.shape = (?,500)
        h_out = output_ + output_masked
        # h_out.shape = (?,500)

        h_out_act = K.tanh(h_out)
        # h_out_act.shape = (?,500)
        # V.shape = (500, 6040, 5)
        # b.shape = (6040,5)
        if self.bias:
            output = tf.tensordot(h_out_act, self.V, axes=[[1], [0]]) + self.b
        else:
            output = tf.tensordot(h_out_act, self.V, axes=[[1], [0]])
        # output.shape = (?,6040,5)
        output = tf.reshape(output, [-1,self.input_dim1,self.input_dim2])
        return output



    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2])