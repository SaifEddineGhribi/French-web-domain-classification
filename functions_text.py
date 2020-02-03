#IMPORTS AND FUNCTION DEFINITIONS

#NLTK

#imports
import nltk #import the natural language toolkit library
from nltk.stem.snowball import FrenchStemmer #import the French stemming library
from nltk.corpus import stopwords #import stopwords from nltk corpus
import re #import the regular expressions library; will be used to strip punctuation
from collections import Counter #allows for counting the number of occurences in a list
import numpy as np
import os #import os module
root_path = "/mnt/c/Users/Saifeddine/Documents/GitHub/data/" #define a working directory path
os.chdir(root_path) #set the working directory path
######################################################
#reading in the raw text from the file
def read_raw_file(path):
    '''reads in raw text from a text file using the argument (path), which represents the path/to/file'''
    f = open(path,"r",errors='ignore') #open the file located at "path" as a file object (f) that is readonly
    raw = f.read()#.decode('utf8') # read raw text into a variable (raw) after decoding it from utf8
    f.close() #close the file now that it isn;t being used any longer
    return raw
######################################################
def get_tokens(raw,encoding='utf8'):
    '''get the nltk tokens from a text'''
    tokens = nltk.word_tokenize(raw) #tokenize the raw UTF-8 text
    return tokens
######################################################
def get_nltk_text(raw,encoding='utf8'):
    '''create an nltk text using the passed argument (raw) after filtering out the commas'''
    #turn the raw text into an nltk text object
    no_commas = re.sub(r'[.|,|\']',' ', raw) #filter out all the commas, periods, and appostrophes using regex
    tokens = nltk.word_tokenize(no_commas) #generate a list of tokens from the raw text
    text=nltk.Text(tokens,encoding) #create a nltk text from those tokens
    return text
######################################################
def get_stopswords(type="veronis"):
    '''returns the veronis stopwords in unicode, or if any other value is passed, it returns the default nltk french stopwords'''
    if type=="veronis":
        #VERONIS STOPWORDS
        raw_stopword_list = ["Ap.", "Apr.", "GHz", "MHz", "USD", "a", "afin", "ah", "ai", "aie", "aient", "aies", "ait", "alors", "après", "as", "attendu", "au", "au-delà", "au-devant", "aucun", "aucune", "audit", "auprès", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "auriez", "aurions", "aurons", "auront", "aussi", "autour", "autre", "autres", "autrui", "aux", "auxdites", "auxdits", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avez", "aviez", "avions", "avons", "ayant", "ayez", "ayons", "b", "bah", "banco", "ben", "bien", "bé", "c", "c'", "c'est", "c'était", "car", "ce", "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci", "celui-là", "celà", "cent", "cents", "cependant", "certain", "certaine", "certaines", "certains", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "cf.", "cg", "cgr", "chacun", "chacune", "chaque", "chez", "ci", "cinq", "cinquante", "cinquante-cinq", "cinquante-deux", "cinquante-et-un", "cinquante-huit", "cinquante-neuf", "cinquante-quatre", "cinquante-sept", "cinquante-six", "cinquante-trois", "cl", "cm", "cm²", "comme", "contre", "d", "d'", "d'après", "d'un", "d'une", "dans", "de", "depuis", "derrière", "des", "desdites", "desdits", "desquelles", "desquels", "deux", "devant", "devers", "dg", "différentes", "différents", "divers", "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept", "dl", "dm", "donc", "dont", "douze", "du", "dudit", "duquel", "durant", "dès", "déjà", "e", "eh", "elle", "elles", "en", "en-dehors", "encore", "enfin", "entre", "envers", "es", "est", "et", "eu", "eue", "eues", "euh", "eurent", "eus", "eusse", "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eûmes", "eût", "eûtes", "f", "fait", "fi", "flac", "fors", "furent", "fus", "fusse", "fussent", "fusses", "fussiez", "fussions", "fut", "fûmes", "fût", "fûtes", "g", "gr", "h", "ha", "han", "hein", "hem", "heu", "hg", "hl", "hm", "hm³", "holà", "hop", "hormis", "hors", "huit", "hum", "hé", "i", "ici", "il", "ils", "j", "j'", "j'ai", "j'avais", "j'étais", "jamais", "je", "jusqu'", "jusqu'au", "jusqu'aux", "jusqu'à", "jusque", "k", "kg", "km", "km²", "l", "l'", "l'autre", "l'on", "l'un", "l'une", "la", "laquelle", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs", "lez", "lors", "lorsqu'", "lorsque", "lui", "lès", "m", "m'", "ma", "maint", "mainte", "maintes", "maints", "mais", "malgré", "me", "mes", "mg", "mgr", "mil", "mille", "milliards", "millions", "ml", "mm", "mm²", "moi", "moins", "mon", "moyennant", "mt", "m²", "m³", "même", "mêmes", "n", "n'avait", "n'y", "ne", "neuf", "ni", "non", "nonante", "nonobstant", "nos", "notre", "nous", "nul", "nulle", "nº", "néanmoins", "o", "octante", "oh", "on", "ont", "onze", "or", "ou", "outre", "où", "p", "par", "par-delà", "parbleu", "parce", "parmi", "pas", "passé", "pendant", "personne", "peu", "plus", "plus_d'un", "plus_d'une", "plusieurs", "pour", "pourquoi", "pourtant", "pourvu", "près", "puisqu'", "puisque", "q", "qu", "qu'", "qu'elle", "qu'elles", "qu'il", "qu'ils", "qu'on", "quand", "quant", "quarante", "quarante-cinq", "quarante-deux", "quarante-et-un", "quarante-huit", "quarante-neuf", "quarante-quatre", "quarante-sept", "quarante-six", "quarante-trois", "quatorze", "quatre", "quatre-vingt", "quatre-vingt-cinq", "quatre-vingt-deux", "quatre-vingt-dix", "quatre-vingt-dix-huit", "quatre-vingt-dix-neuf", "quatre-vingt-dix-sept", "quatre-vingt-douze", "quatre-vingt-huit", "quatre-vingt-neuf", "quatre-vingt-onze", "quatre-vingt-quatorze", "quatre-vingt-quatre", "quatre-vingt-quinze", "quatre-vingt-seize", "quatre-vingt-sept", "quatre-vingt-six", "quatre-vingt-treize", "quatre-vingt-trois", "quatre-vingt-un", "quatre-vingt-une", "quatre-vingts", "que", "quel", "quelle", "quelles", "quelqu'", "quelqu'un", "quelqu'une", "quelque", "quelques", "quelques-unes", "quelques-uns", "quels", "qui", "quiconque", "quinze", "quoi", "quoiqu'", "quoique", "r", "revoici", "revoilà", "rien", "s", "s'", "sa", "sans", "sauf", "se", "seize", "selon", "sept", "septante", "sera", "serai", "seraient", "serais", "serait", "seras", "serez", "seriez", "serions", "serons", "seront", "ses", "si", "sinon", "six", "soi", "soient", "sois", "soit", "soixante", "soixante-cinq", "soixante-deux", "soixante-dix", "soixante-dix-huit", "soixante-dix-neuf", "soixante-dix-sept", "soixante-douze", "soixante-et-onze", "soixante-et-un", "soixante-et-une", "soixante-huit", "soixante-neuf", "soixante-quatorze", "soixante-quatre", "soixante-quinze", "soixante-seize", "soixante-sept", "soixante-six", "soixante-treize", "soixante-trois", "sommes", "son", "sont", "sous", "soyez", "soyons", "suis", "suite", "sur", "sus", "t", "t'", "ta", "tacatac", "tandis", "te", "tel", "telle", "telles", "tels", "tes", "toi", "ton", "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente", "trente-cinq", "trente-deux", "trente-et-un", "trente-huit", "trente-neuf", "trente-quatre", "trente-sept", "trente-six", "trente-trois", "trois", "très", "tu", "u", "un", "une", "unes", "uns", "v", "vers", "via", "vingt", "vingt-cinq", "vingt-deux", "vingt-huit", "vingt-neuf", "vingt-quatre", "vingt-sept", "vingt-six", "vingt-trois", "vis-à-vis", "voici", "voilà", "vos", "votre", "vous", "w", "x", "y", "z", "zéro", "à", "ç'", "ça", "ès", "étaient", "étais", "était", "étant", "étiez", "étions", "été", "étée", "étées", "étés", "êtes", "être", "ô"]
    else:
        #get French stopwords from the nltk kit
        raw_stopword_list = stopwords.words('french') #create a list of all French stopwords
    #stopword_list = [word.decode('utf8') for word in raw_stopword_list] #make to decode the French stopwords as unicode objects rather than ascii
    stopword_list = [word for word in raw_stopword_list] 
    return stopword_list
 ######################################################   

def filter_stopwords(text,stopword_list):
    '''normalizes the words by turning them all lowercase and then filters out the stopwords'''
    words=[w.lower() for w in text] #normalize the words in the text, making them all lowercase
    #filtering stopwords
    filtered_words = [] #declare an empty list to hold our filtered words
    for word in words: #iterate over all words from the text
        if word not in stopword_list and word.isalpha() and len(word) > 1: #only add words that are not in the French stopwords list, are alphabetic, and are more than 1 character
            filtered_words.append(word) #add word to filter_words list if it meets the above conditions
    filtered_words.sort() #sort filtered_words list
    return filtered_words
 ######################################################   
def stem_words(words):
    '''stems the word list using the French Stemmer'''
    #stemming words
    stemmed_words = [] #declare an empty list to hold our stemmed words
    stemmer = FrenchStemmer() #create a stemmer object in the FrenchStemmer class
    for word in words:
        stemmed_word=stemmer.stem(word) #stem the word
        stemmed_words.append(stemmed_word) #add it to our stemmed word list
    stemmed_words.sort() #sort the stemmed_words
    return stemmed_words
######################################################   
def sort_dictionary(dictionary):
    '''returns a sorted dictionary (as tuples) based on the value of each key'''
    return sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
######################################################
def normalize_counts(counts):
    total = sum(counts.values())
    return dict((word, float(count)/total) for word,count in counts.items())
 ######################################################       
def print_sorted_dictionary(tuple_dict):
    '''print the results of sort_dictionary'''
    for tup in tuple_dict:
        print (unicode(tup[1])[0:10] + '\t\t' + unicode(tup[0]))
######################################################        
def print_words(words):
    '''clean print the unicode words'''
    for word in words:
        print( word )
######################################################        
def pre_process(text,max_words = 100):    
    tokens = get_tokens(text)
    nltk_text = get_nltk_text(text)
    stop_words = get_stopswords()    
    filtered_words = filter_stopwords(nltk_text,stop_words)
    stemmed_words = stem_words(filtered_words)
    #np.shape(np.unique(stemmed_words))
    dict_words = Counter(stemmed_words)
    dict_words = normalize_counts(dict_words)
    dict_words = list(sort_dictionary(dict_words)[:max_words])   
    if len(dict_words)==0:
        return -1
    x = np.array(dict_words)[:,0]
    return x 

######################################################

# a function to transform vector to one-hot encoding 
def transform_one_hot(y):
    m,n = y.shape[0], np.max(y)+1
    Y_one_hot_train = np.zeros((m,n))
    for i in range (m):
        Y_one_hot_train[i,y[i]]=1
    return Y_one_hot_train

######################################################

#a functions that from tokens returns word_to_id, id_to_word
def mapping(tokens):
    word_to_id = dict()
    id_to_word = dict()

    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token

    return word_to_id, id_to_word

######################################################
def encode(X):
    #a function that encodes words w.r.t. to their id in word_to_id dictionary
    res = list()
    for i in X :
        temp  = list()
        for j in i :
            temp.append(word_to_id[j])
        res.append(temp)        
    return res

def get_data (keys,labels,stop_before = -1,max_words=100,show_every= 100):
    k = 1   
    ids,X,y = [],[],[]
    for (key,label) in zip(keys,labels):
        root = "/mnt/c/Users/Saifeddine/Documents/GitHub/data/text/text2/"
        directory = root+str(key)    
        file = read_raw_file(directory)
        file_vector = pre_process(file,max_words =max_words)
        if (file_vector!=-1):
            X.append(file_vector)
            y.append(label)
            ids.append(key)
        else : 
            print("found blank file on "+ str(key))
        k+=1
        if (k%show_every==0): 
            print(str(k) + " files processed")

        if (k > stop_before & stop_before !=-1):
            break  
    return ids,X,y
################################################
#get test data
def get_test (test,max_words= 100):
    root = "/mnt/c/Users/Saifeddine/Documents/GitHub/data/text/text2/"
    X=[]
    k=1
    for key in list(test["text"]):
        directory = root+str(key)    
        file = read_raw_file(directory)
        file_vector = pre_process(file,max_words =max_words)
        if (file_vector!=-1):
            X.append(file_vector)
        k+=1
        if (k%20==0): 
            print(str(k) + " files processed")
    return np.array(X)

###############################################
def get_vects(ids,X,y):
    Y_one_hot_train = transform_one_hot(y)
    k = 0
    XX_train = list()
    yy_train = list()
    ids_train =[]
    for i in list(X.ravel()):
        if (k>= Y_one_hot_train.shape[0]):
            break
        if (len(i)>=80): 
            XX_train.append(i[:80])
            yy_train.append(Y_one_hot_train[k])
            ids_train.append(ids[k])
        
        k+=1
    XX_train = np.array(XX_train)
    yy_train = np.array(yy_train)
    return XX_train,yy_train
#USING STANFORD'S FRENCH POS TAGGER, v.3.2
#http://nlp.stanford.edu/software/tagger.shtml
#note: to get NLTK to find java with the tagger, I had to comment out lines 59 and 85 [config_java(options=self.java_options, verbose=False)] in stanford.py [C:\Anaconda\Lib\site-packages\nltk\tag\stanford.py]
#then I had to set the python path directly
        
import nltk #import the Natural Language Processing Kit
