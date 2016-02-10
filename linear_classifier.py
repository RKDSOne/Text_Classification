import numpy as np


class LinearClassifier():

    def __init__(self):
        self.trained = False

    def train(self,x,y):
        '''
        Returns the weight vector
        '''
        raise NotImplementedError('LinearClassifier.train not implemented')

    def get_scores(self,x,w):
        '''
        Computes the dot product between X,w
        '''
        return np.dot(x,w)

    def get_label(self,x,w):
        '''
        Computes the label for each data point
        '''
        scores = np.dot(x,w)
        return np.argmax(scores,axis=1).transpose()

    def test(self,x,w):
        '''
        Classifies the points based on a weight vector.
        '''
        if self.trained == False:
            raise ValueError("Model not trained. Cannot test")
            return 0
        x = self.add_intercept_term(x)
        return self.get_label(x,w)
    
    def add_intercept_term(self,x):
        ''' Adds a column of ones to estimate the intercept term for separation boundary'''
        nr_x, nr_f = x.shape
        intercept = np.ones([nr_x,1])
        x = np.hstack((intercept,x))
        return x

    def evaluate(self,truth,predicted):
        correct_g,wrong_g,total_g = 0.0,0.0,0.0
        correct_a,wrong_a,total_a = 0.0,0.0,0.0
        correct_n,wrong_n,total_n = 0.0,0.0,0.0
        
             
        for i in range(len(truth)):
            if truth[i] == 0:
                total_g += 1                
                if (predicted[i] == 0):                    
                    correct_g += 1
                if (predicted[i] == 1):
                    wrong_a += 1
                if (predicted[i] == 2):
                    wrong_n += 1 
              
            if truth[i] == 1:
                total_a += 1                
                if (predicted[i] == 0):
                    wrong_g += 1
                if(predicted[i] == 1):
                    correct_a += 1
                if (predicted[i] == 2):
                    wrong_n += 1
              
            if truth[i] == 2:            
                total_n += 1                
                if(predicted[i] == 0):
                   wrong_g += 1                
                if(predicted[i] == 1):
                   wrong_a += 1 
                if(predicted[i] == 2):
                    correct_n += 1
                
        f_g = total_g - correct_g
        f1_g = 2 * correct_g /(2*correct_g + f_g + wrong_g)
        f_a = total_a - correct_a
        f1_a = 2 * correct_a /(2*correct_a + f_a + wrong_a)
        f_n = total_n - correct_n
        f1_n = 2 * correct_n /(2*correct_n + f_n + wrong_n)		
        MavgF1 = (f1_g + f1_a + f1_n) / 3
        
                		
        return MavgF1
