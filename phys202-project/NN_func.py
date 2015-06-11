import numpy as np

from sklearn.datasets import load_digits
digits = load_digits()

def sigmoid(x):
    return 1/(1+np.exp(-x))                     #Activation function

def network_setup(ni, k, npl, no):
    
    """
    ni : int, number of neurons in input layer
    
    k : int, number of layers (excluding input and output layer)
    
    npl : int, number of neurons per hidden layer
    
    no : int, number of neurons in output layer
    
    *************************************
    
    returns
    
    w : list of arrays, weights between each neuron
    
    b : list, biases of each neuron
    """
    
#     np.random.seed(0)                           #For testing.
    k = int(k)
    npl = int(npl)
    
    w = []                                      #Weights
    w0 = np.random.randn(ni,npl)                #First weight, all 1's
    w.append(w0)
    if k > 1:
        for i in range(k-1):
            w.append(np.random.randn(npl,npl))
    w.append(np.random.randn(npl,no))           #Last weight, 10 outputs
    
    b = []                                      #Biases
    for i in range(k):
        b.append(np.random.randn(1,npl))
    b.append(np.random.randn(1,no))             #Last biases, 10 outputs
    
    return w,b

def feedforward(w,b,x,k):
    
    """
    w, b : weights and biases from network_setup
    
    x : input neuron values
    
    k : int, number of hidden layers
    
    **************************************
    
    returns
    
    z : array of activation function inputs
    
    a : array of activation function outputs
    """
    
    z = []                                                  #Array of activation function inputs
    a = []                                                  #Initialize array for activation function values
    a.append(x)
    for i in range(k+1):                                    #k+1 involves hidden layers plus output layer
        zi = []
        for j in range(len(b[i][0,:])):
            zi.append(np.sum(w[i][:,j]*a[i]) + b[i][0,j])   #Loop through each weight and bias array to get weighted inputs of all neurons in the next layer.
        zz = np.array(zi)
        z.append(zz)
        a.append(sigmoid(zz))                               #Array of outputs
    return z, a

def cost(n, y, a):
    
    """
    n : int, total number of training examples
    
    y : array, desired output
    
    a : list of arrays, output from activation functions
    
    ***********************************
    
    returns
    
    c : float, cost function value
    """
    
    return (1/(2*n))*np.sum((y-a)**2)               # Cost function

def sigmoid_prime(x):
    return np.exp(-x)/((1+np.exp(-x))**2)           # First derivative of sigmoid function

sigmoid_prime_vec = np.vectorize(sigmoid_prime)

def act_error(a, y, z, w, k):
    
    """
    a : output from network
    
    y : desired output
    
    z : weighted input array
    
    w : weights list of arrays
    
    **********************************
    
    returns
    
    delta : list of arrays, error from each from neuron
    """
    
    delta = []                                      #Sets up delta list
    y_con = np.zeros(10)
    y_con[y] = 1                                    #Converts a digit 0-9 into the network output format
    
    d = (a[-1] - y_con)*sigmoid_prime(z[-1])                                 #Calculate error for last neuron layer
    delta.append(d)                                                             #Append it to delta list
    for i in range(k):                                                          #Cycle through each other layer...
        di = np.dot(w[-(i+1)],delta[0])*sigmoid_prime(z[-(i+2)])             #...in reverse order
        delta.insert(0,di)
    
    return delta

def replace(w,b,eta,m,delta,a):
    
    """
    w, b : weights and biases
    
    eta : float, learning rate
    
    m : int, size of mini batch
    
    delta : list of arrays, error from each from neuron
    
    a : output from network
    
    ***************************************
    
    w, b : new weights and biases
    """
    
    for i in range(len(w)):
        w[-(i+1)] = w[-(i+1)] - (eta/m)*np.sum(np.dot(np.transpose(np.column_stack(delta[-(i+1)])),np.column_stack(a[-(i+2)])))         #Equations from Michael Nielsen article
        b[-(i+1)] = b[-(i+1)] - (eta/m)*np.sum(delta[-(i+1)])
        
    return w, b

def create_sets(n):
    
    """    
    n : int, number of terms in training set
    
    *****************************
    
    returns
    
    training_set : dict, contains 'data' (list of arrays) and 'target' (list) where 
                   each target corresponds to a certain data of the same index.
                   
    test_set : dict, contains 'data' (list of arrays) and 'target' (list) where 
               each target corresponds to a certain data of the same index.
    """
    
    indices = np.linspace(0,len(digits.target)-1,len(digits.target), dtype='int')   #Specialized for sklearn data set
    np.random.shuffle(indices)                                                      #Randomize indices for random data set

    training_set = {'data': [],'target': []}                                        #Creates training set dictionary
    for i in range(n):
        training_set['data'].append(digits.data[indices[i]])
        training_set['target'].append(digits.target[indices[i]])

    test_set = {'data': [],'target': []}                                            #Creates test set dictionary
    for j in range(len(digits.target)-n):
        test_set['data'].append(digits.data[indices[-(j+1)]])
        test_set['target'].append(digits.target[indices[-(j+1)]])

    return training_set, test_set

def mini_batch(m, training_set):
    
    """
    m : int, number of terms in mini batch
    
    training_set : dict, from result of create_sets
    
    ************************************
    
    returns
    
    mini : list of dicts, mini batches to be used for training.
           Callings for the mini batches work as follows:
                   
                   mini[j][k][l]
                   
           where j corresponds to a mini batch, k can be either 
           'data' or 'target', and l corresponds to a particular
           digit.
    """
    
    n = len(training_set['target'])                                 #Number of elements in training set
    m_indices = np.linspace(0,n-1,n, dtype='int')                   #All indices of the training set
    np.random.shuffle(m_indices)                                    #Randomize their order
    
    mini = []                                                       #Establish the 'mini' list
    mini_in = []                                                    #List for the randomized index arrays of the mini
    for i in range(int(n/m)):                                       #Creates ~n/m mini batches
        mini_in.append(m_indices[i*m:(i+1)*m])                      #Mini batches of size m
                                                                        #Now translate the indices into actual data
    for j in range(len(mini_in)):                                                   #For each mini batch...
        mini_dat_sub = {'data': [], 'target': []}                                   #Create intermediate dict
        for k in range(len(mini_in[j])):                                            #For each element in mini batch... 
            mini_dat_sub['data'].append(training_set['data'][mini_in[j][k]])        #Append data to intermediate dict
            mini_dat_sub['target'].append(training_set['target'][mini_in[j][k]])    #Append target to intermediate dict
        mini.append(mini_dat_sub)                                                   #Append dicts to 'mini' list
        
    return mini

def neural_network_super_function(ni,k,npl,no,eta,m,n,epochs):
    
    """
    Combines all worthwhile aspects of previously stated functions
    to create a learning system.
    
    epochs : int, number of times a new batch is randomly created
             to run through.
             
    ***********************************************
    
    returns
    
    w, b : weights and biases that have been trained.
    """
    
    training_set, test_set = create_sets(n)                      #Gimme data!
    
    w, b = network_setup(ni,k,npl,no)                            #Establish neurons and their connections
    
    for r in range(epochs):                                         #For each epoch...
        mini = mini_batch(m,training_set)                        #Prepping mini batch
        for i in range(len(mini)):                                  #For each mini batch...
            for j in range(len(mini[i]['target'])):                 #For each digit in each mini batch...
                x = mini[i]['data'][j]                              #Define x as data input
                y = mini[i]['target'][j]                            #Define y as desired output
            
                z, a = feedforward(w,b,x,k)                      #Find the weighted inputs and activation function returns

                delta = act_error(a,y,z,w,k)                     #Find the error in each neuron
            
                w, b = replace(w,b,eta,m,delta,a)                #Define the new weights and biases!
        print('Epoch {0} Complete'.format(r+1))

    return w, b, test_set

def test_net(w,b,test_set,k):
    
    """
    w, b : weights and balances of trained network
    
    test_set : test set from create_sets function
    
    k : number of hidden layers (for feedforward func)
    
    ******************************************
    
    prints statement
    """
    
    count = 0
    for i in range(len(test_set['target'])):
        x = test_set['data'][i]
        y = test_set['target'][i]
        y_con = np.zeros(10)
        y_con[y] = 1
        z, a = feedforward(w,b,x,k)
        
        out = a[-1]
        max_index = 0
        for j in range(len(out)):
            if out[j] > max_index:
                max_index = out[j]
        for p in range(len(out)):
            if out[p] == max_index:
                out[p] = 1
            else:
                out[p] = 0
                
        correct_indi = 0
        for q in range(len(out)):
            if out[q] == y_con[q]:
                correct_indi += 1
                if correct_indi == 10:
                    count += 1
                
    print('Accuracy: {0}/{1}'.format(count,len(test_set['target'])))
    return a, out, y_con, w, b