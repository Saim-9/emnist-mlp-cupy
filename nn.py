import cupy as cp


class Layer_Dense:

    def __init__(self, n_features, n_neurons, weight_reg_l1=0, weight_reg_l2=0, 
                 bias_reg_l1=0, bias_reg_l2=0):
 

        self.weights = cp.random.randn(n_features, n_neurons, dtype=cp.float32) *\
            (cp.sqrt(cp.array(2.0/n_features, dtype=cp.float32)))
        self.biases = cp.zeros((1, n_neurons), dtype=cp.float32)

        self.weight_reg_l1 = weight_reg_l1
        self.weight_reg_l2 = weight_reg_l2
        self.bias_reg_l1 = bias_reg_l1
        self.bias_reg_l2 = bias_reg_l2


    def forward(self, inputs):
        self.inputs = inputs                                    # saving as will be used in back prop for calculting derv of weight
        self.output = (cp.dot(inputs, self.weights)) + self.biases


    def backward(self, dvalues):
        self.dinputs = cp.dot(dvalues, self.weights.T)
        self.dweights = cp.dot(self.inputs.T, dvalues)    
        self.dbiases = cp.sum(dvalues, axis=0, keepdims=True) 


        if self.weight_reg_l1 > 0:
            self.dweights += self.weight_reg_l1 * cp.sign(self.weights)

        if(self.weight_reg_l2 > 0):
            self.dweights += 2 * self.weight_reg_l2 * self.weights 


        if self.bias_reg_l1 > 0:
            self.dbiases += self.bias_reg_l1 * cp.sign(self.biases)


        if(self.bias_reg_l2 > 0):
            self.dbiases += 2 * self.bias_reg_l2 * self.biases             
           

class Layer_Dropout:

    def __init__(self, rate):   #rate means the rate of dropout (exclusion)
        self.rate = 1 - rate    # we convert the dropout rate to the amount of neurons we want to have 

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = cp.random.binomial(1, self.rate, size=inputs.shape)\
            .astype(cp.float32) / (self.rate)
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs =dvalues * self.binary_mask    
        


class Activation_ReLU:
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output =  cp.maximum(0, inputs)    

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.output <= 0] = 0           


class Activation_Softmax:
        
    def forward(self, inputs):
        exp_values = cp.exp(inputs - cp.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / cp.sum(exp_values, axis=1, keepdims=True)


    def backward(self, dvalues):

        self.dinputs = cp.empty_like(dvalues, dtype=cp.float32)

        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):

            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = cp.diagflat(single_output) - cp.dot(single_output, single_output.T)
            self.dinputs[index] = cp.dot(jacobian_matrix, single_dvalues)
                

class Loss:

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = cp.mean(sample_losses)
        return data_loss

    def regularization_loss(self, layer):

        regularization_loss = 0

        if(layer.weight_reg_l1 > 0):
            regularization_loss += layer.weight_reg_l1 * (cp.sum(cp.abs(layer.weights)))

        if(layer.weight_reg_l2 > 0):
            regularization_loss += layer.weight_reg_l2 * (cp.sum(layer.weights * layer.weights))

        if(layer.bias_reg_l1 > 0):
            regularization_loss += layer.bias_reg_l1 * (cp.sum(cp.abs(layer.biases)))


        if(layer.bias_reg_l2 > 0):
            regularization_loss += layer.bias_reg_l2 * (cp.sum(layer.biases*layer.biases))      

        return regularization_loss                      



class Loss_Categorical_Crossentropy(Loss):

    def forward(self, y_pred, y_true_classes):
        y_clipped = cp.clip(y_pred, 1e-7, 1 - 1e-7)

        if(y_true_classes.ndim == 1):
            y_results = y_clipped[cp.arange(y_clipped.shape[0]), y_true_classes]

        elif(y_true_classes.ndim == 2):   #if onehot encoded
            y_results = cp.sum(y_clipped * y_true_classes, axis=1)    

        nlog_vals = -cp.log(y_results)
        return nlog_vals
    

    def backward(self, dvalues, y_true):    #dvalues are our y_preds
        n_samples = len(dvalues)            # no of samples in this batch
        n_labels = len(dvalues[0])    # no of labels in one sample, need this to one hot encode if it isn't

        if(len(y_true.shape) == 1):
            y_true = cp.eye(n_labels, dtype=cp.float32)[y_true]

        dvalues = cp.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -y_true/dvalues   

        # for normlaizing gradients
        self.dinputs = self.dinputs / n_samples  



class Activation_Softmax_Loss_CategoricalCrossentropy(): 
 
    def __init__(self): 
        self.activation = Activation_Softmax() 
        self.loss = Loss_Categorical_Crossentropy() 
 
 
    def forward(self, inputs, y_true): 

        self.activation.forward(inputs)
        self.output = self.activation.output 
        return self.loss.calculate(self.output, y_true) 
 

    def backward(self, dvalues, y_true): 
 
        samples = len(dvalues) 
        if len(y_true.shape) == 2: 
            y_true = cp.argmax(y_true, axis=1) 
 
        self.dinputs = dvalues.copy()
        self.dinputs[cp.arange(samples), y_true] -= 1

        # Normalizing gradient 
        self.dinputs = self.dinputs / samples

class Optimizer_SGD():

    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum


    def pre_update_params(self):
        if (self.decay):
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + (self.decay*self.iterations)))


    def update_params(self, layer):

        if(self.momentum):

            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = cp.zeros_like(layer.weights)
                layer.bias_momentums = cp.zeros_like(layer.biases)

            weight_updates = (self.momentum*layer.weight_momentums) -\
                (self.current_learning_rate*layer.dweights) 
            layer.weight_momentums = weight_updates

            bias_updates = (self.momentum*layer.bias_momentums) -\
                (self.current_learning_rate*layer.dbiases)    
            layer.bias_momentums = bias_updates

        else:

            weight_updates = -self.current_learning_rate*layer.dweights     
            bias_updates = -self.current_learning_rate*layer.dbiases


        layer.weights += weight_updates
        layer.biases += bias_updates    


    def post_update_params(self):
        self.iterations += 1         



class Optimizer_Adam():


    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2



    def pre_update_params(self):
        if (self.decay):
            self.current_learning_rate = self.learning_rate*(1. / (1. + (self.decay*self.iterations)))


    def update_params(self, layer):        

        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = cp.zeros_like(layer.weights)
            layer.weight_cache = cp.zeros_like(layer.weights)
            
            layer.bias_momentums = cp.zeros_like(layer.biases)
            layer.bias_cache = cp.zeros_like(layer.biases)


        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
                                    (1 - self.beta_1) * layer.dweights
        
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
                                    (1 - self.beta_1) * layer.dbiases
        
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / \
                        (cp.sqrt(weight_cache_corrected) + self.epsilon) 
        
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
                        (cp.sqrt(bias_cache_corrected) + self.epsilon)         
   


    def post_update_params(self):
        self.iterations += 1 



        
