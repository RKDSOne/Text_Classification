# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import nltk
import re
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import codecs
import time
start = time.time()
lemma_word = nltk.WordNetLemmatizer() # calling the lemmatization module
elimi = set(stopwords.words('english')) # Parameter for stopwords removal
elimi.update(['b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
# Updating the stopwords list
def find_vocab( # The function is used to update our vocabulary 
vocab, # stores the words
gcontent, # Stores the content of each file for a category
gwords): # Stores the words from each file
    for i in range(len(gcontent)):
        gcontent[i] = re.sub('[^A-Za-z]|[!,@,#,$,%,^,&,*,?,_,~,-,£,(,)]',' ', gcontent[i]) # eliminating special characters, numbers 
        gcontent[i] = re.sub('\W+',' ', gcontent[i]) # replacing white spaces with single space.
        gcontent[i] = gcontent[i].lower() #replacing caps with small letters
        gwords.append(word_tokenize(gcontent[i])) # word tokanization    
        for j in range(len(gwords[i])):
            gwords[i][j] = lemma_word.lemmatize(gwords[i][j]) # lemmatizing words
        gwords[i] = [k for k in gwords[i] if k not in elimi] # stopwords removal
        gwords[i] = list(set(gwords[i]))        
        for j in range(len(gwords[i])):            
            if not gwords[i][j] in vocab:
                vocab.append(gwords[i][j]) # updating the vocab
    return gwords

def preprocess_text ( # The function is used to write the preprocessed data in a file
gwords # the words from a category
,category): # The category of the data being used
    for i in range(len(gwords)):
        for j in range(len(gwords[i])):
            file.write(str(gwords[i][j]) + ':' +str(gwords[i].count(gwords[i][j])) + ' ')
            # writing the words in a filw with a proper format
        file.write('#label#:'+ str(category) +'\n')
        # adding label at the end of each file
    file.close()

def create_content (gdocs,graphicsdir,gcontent):
    for file in gdocs:
        gcontent.append(open(graphicsdir+'/'+str(file),'r').read())


# defining the directory path for each category
graphicsdir,autosdir,gunsdir = '20news-bydate/train/comp.graphics','20news-bydate/train/rec.autos','20news-bydate/train/talk.politics.guns'
graphicstest,autostest,gunstest = '20news-bydate/test/comp.graphics','20news-bydate/test/rec.autos','20news-bydate/test/talk.politics.guns'
graphicscorpus,autoscorpus,gunscorpus = PlaintextCorpusReader(graphicsdir, '.*'),PlaintextCorpusReader(autosdir, '.*'),PlaintextCorpusReader(gunsdir, '.*')
graphicscorpustest,autoscorpustest,gunscorpustest = PlaintextCorpusReader(graphicstest, '.*'),PlaintextCorpusReader(autostest, '.*'),PlaintextCorpusReader(gunstest, '.*')

# initializing the lists
gdocs,adocs,ndocs,gcontent,acontent,ncontent,gwords,awords,nwords,vocab = [],[],[],[],[],[],[],[],[],[]
gtdocs,atdocs,ntdocs,gtcontent,atcontent,ntcontent,gtwords,atwords,ntwords,vtocab = [],[],[],[],[],[],[],[],[],[]
# for train dataset
gdocs.extend(graphicscorpus.fileids()) # for graphics category
adocs.extend(autoscorpus.fileids()) # for autos category
ndocs.extend(gunscorpus.fileids()) # for guns category
# for test dataset
gtdocs.extend(graphicscorpustest.fileids()) # for graphics category
atdocs.extend(autoscorpustest.fileids()) # for autos category
ntdocs.extend(gunscorpustest.fileids()) # for guns category
# retriving the words for each category
# for train dataset
create_content(gdocs,graphicsdir,gcontent)
create_content(adocs,autosdir,acontent)
create_content(ndocs,gunsdir,ncontent)
# for test dataset
create_content(gtdocs,graphicstest,gtcontent)
create_content(atdocs,autostest,atcontent)
create_content(ntdocs,gunstest,ntcontent)
# Finding the vocab in each category
gwords = find_vocab(vocab,gcontent,gwords)
awords = find_vocab(vocab,acontent,awords)
nwords = find_vocab(vocab,ncontent,nwords)
print 'The number of unique words in all 3 categories of training dataset is:', (len(vocab))
gtwords = find_vocab(vtocab,gtcontent,gtwords)
atwords = find_vocab(vtocab,atcontent,atwords)
ntwords = find_vocab(vtocab,ntcontent,ntwords)
# writing the preprocessed text in different files
file = codecs.open("./graphics.train",'w','UTF8')
preprocess_text(gwords,'graphics')
file = codecs.open("./autos.train",'w','UTF8')
preprocess_text(awords,'autos')
file = codecs.open("./guns.train",'w','UTF8')
preprocess_text(nwords,'guns')
# for test dataset
file = codecs.open("./graphics.test",'w','UTF8')
preprocess_text(gtwords,'graphics')
file = codecs.open("./autos.test",'w','UTF8')
preprocess_text(atwords,'autos')
file = codecs.open("./guns.test",'w','UTF8')
preprocess_text(ntwords,'guns')

print "TIme taken to execute the program is: %f seconds" % (int(time.time() - start))