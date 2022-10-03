#  Biblioteca de preprocesamiento de datos de texto
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# Para las palabras raíz
from nltk.stem import PorterStemmer

# Crear una instancia para la clase PorterStemmer
stemmer = PorterStemmer()

# Importar la biblioteca json
import json
import pickle
import numpy as np

words=[]
classes = []
word_tags_list = []
ignore_words = ['?', '!',',','.', "'s", "'m"]
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)

# Función para añadir palabras raíz (stem words)
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)  
    return stem_words

for intent in intents['intents']:
    
        # Agregar todas las palabras de los patrones a una lista
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                      
            word_tags_list.append((pattern_word, intent['tag']))
        # Agregar todas las etiquetas a la lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            stem_words = get_stem_words(words, ignore_words)
print("Palabras RAIZ")
print(stem_words)
print("Lista de PAREJAS")
print(word_tags_list[0]) 
print("CLASES")
print(classes)   

# Crear un corpus de palabras para el chatbot
def create_bot_corpus(stem_words, classes):

    #sorted ordena alfabeticamente
    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    #dump crea 2 documentos con listas
    #wb significa escritura en forma binaria
    pickle.dump(stem_words, open('words.pkl','wb'))
    pickle.dump(classes, open('classes.pkl','wb'))

    return stem_words, classes

stem_words, classes = create_bot_corpus(stem_words,classes)  

print("En orden alfabetico")
print(stem_words)
print(classes)

# Crear una bolsa de palabras
bag=[]
num_clases=len(classes)
lista_ceros=[0]*num_clases

for cero in word_tags_list:
    bag_words=[]
    pattern_words=cero[0]

    for x in pattern_words:
        index=pattern_words.index(x)
        x = stemmer.stem(x.lower())
        pattern_words[index]=x

    for z in stem_words:
        if x in pattern_words:
            bag_words.append(1)
        else:
            bag_words.append(0)
    print(bag_words)
    
    labels_incoding=list(lista_ceros)
    tag=cero[1]
    tag_index=classes.index(tag)
    labels_incoding[tag_index]=1
    bag.append([bag_words,labels_incoding])

print(bag[0])

