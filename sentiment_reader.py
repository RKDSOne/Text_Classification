import codecs
import numpy as np

class SentimentCorpus:
    
    def __init__(self, train_per=1.0, dev_per=0, test_per=0.0):
        '''
        prepare dataset
        1) build feature dictionaries
        2) split data into train/dev/test sets 
        '''
        train_X, train_y,test_X, test_y, feat_dict, feat_counts = build_dicts()
        self.nr_instances = train_y.shape[0]
        self.nr_features = train_X.shape[1]
        self.train_X = train_X
        self.train_y = train_y
        self.feat_dict = feat_dict
        self.feat_counts = feat_counts
        self.test_X = test_X
        self.test_y = test_y

        
        

def split_train_dev_test(X, y, train_per, dev_per, test_per):
    if (train_per + dev_per + test_per) > 1:
        print "train/dev/test splits should sum to one"
        return
    dim = y.shape[0]
    split1 = int(dim * train_per)
    
    if dev_per == 0:
        train_y, test_y = np.vsplit(y, [split1])
        dev_y = np.array([])
        train_X = X[0:split1,:]
        test_X = X[split1:,:]
        dev_X = np.array([])
    else:
        split2 = int(dim*(train_per+dev_per))
        train_y,dev_y,test_y = np.vsplit(y,(split1,split2))
        train_X = X[0:split1,:]
        dev_X = X[split1:split2,:]
        test_X = X[split2:,:]
        
    return train_y,dev_y,test_y,train_X,dev_X,test_X

def build_dicts():
    '''
    builds feature dictionaries
    ''' 
    feat_counts = {}

    # build feature dictionary with counts
    nr_g = 0
    with codecs.open("graphics.train", 'r', 'utf8') as g_file:
        for line in g_file:
            nr_g += 1
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name not in feat_counts:
                    feat_counts[name] = 0
                feat_counts[name] += int(counts)
    
    nr_a = 0
    with codecs.open("autos.train", 'r', 'utf8') as n_file:
        for line in n_file:
            nr_a += 1
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name not in feat_counts:
                    feat_counts[name] = 0
                feat_counts[name] += int(counts)
	
	nr_n = 0
    with codecs.open("guns.train", 'r', 'utf8') as n_file:
        for line in n_file:
            nr_n += 1
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name not in feat_counts:
                    feat_counts[name] = 0
                feat_counts[name] += int(counts)

    # remove all features that occur less than 5 (threshold) times
    to_remove = []
    for key, value in feat_counts.iteritems():
        if value < 5:
            to_remove.append(key)
    for key in to_remove:
        del feat_counts[key]

    # map feature to index
    feat_dict = {}
    i = 0
    for key in feat_counts.keys():
        feat_dict[key] = i
        i += 1

    nr_feat = len(feat_counts) 
    nr_instances = nr_g + nr_a + nr_n
    X = np.zeros((nr_instances, nr_feat), dtype=float)
    y = np.vstack((np.zeros([nr_g,1], dtype=int), np.ones([nr_a,1], dtype=int), (2 * np.ones([nr_n,1], dtype=int))))
    
    with codecs.open("graphics.train", 'r', 'utf8') as g_file:
        nr_g = 0
        for line in g_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    X[nr_g,feat_dict[name]] = int(counts)
            nr_g += 1
        
    with codecs.open("autos.train", 'r', 'utf8') as n_file:
        nr_a = 0
        for line in n_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    X[nr_g+nr_a,feat_dict[name]] = int(counts)
            nr_a += 1
    with codecs.open("guns.train", 'r', 'utf8') as n_file:
        nr_n = 0
        for line in n_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    X[nr_g+nr_a+nr_n,feat_dict[name]] = int(counts)
            nr_n += 1

#testing dataset
    ts_g = 0
    with codecs.open("graphics.test" , 'r', 'utf8') as g_file:
        for line in g_file:            
            ts_g += 1

    ts_a = 0
    with codecs.open("autos.test", 'r', 'utf8') as a_file:
        for line in a_file:
            ts_a += 1

    ts_n = 0
    with codecs.open("guns.test", 'r', 'utf8') as n_file:
        for line in n_file:
            ts_n += 1
	
    ts_instances = ts_g + ts_a + ts_n
    test_X = np.zeros((ts_instances, nr_feat), dtype=float)
    test_y = np.vstack(( np.zeros([ts_g,1], dtype=int), np.ones([ts_a,1], dtype=int), 2 * np.ones([ts_n,1], dtype=int) )) 

    with codecs.open("graphics.test", 'r', 'utf8') as g_file:        
        ts_g = 0
        for line in g_file:
            toks = line.split(" ")            
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    test_X[ts_g,feat_dict[name]] = int(counts)
            ts_g += 1
        
    with codecs.open("autos.test", 'r', 'utf8') as a_file:
        ts_a = 0
        for line in a_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    test_X[ts_g+ts_a,feat_dict[name]] = int(counts)
            ts_a += 1

    with codecs.open("guns.test", 'r', 'utf8') as n_file:
        ts_n = 0
        for line in n_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    test_X[ts_g+ts_a+ts_n,feat_dict[name]] = int(counts)
            ts_n += 1   
    
    # shuffle the order, mix all three categories
	# for training dataset
    new_order = np.arange(nr_instances)
    np.random.seed(0) # set seed
    np.random.shuffle(new_order)
    X = X[new_order,:]
    y = y[new_order,:]
    ''' for testing dataset
    new_order_t = np.arange(ts_instances)
    np.random.seed(0) # set seed
    np.random.shuffle(new_order_t)
    test_X = test_X[new_order_t,:]
    test_y = test_y[new_order_t,:]'''
    
    return X, y,test_X, test_y, feat_dict, feat_counts,