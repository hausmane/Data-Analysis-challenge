import math
import gensim
import json


# Read the json file
with open('/home/user/Téléchargements/DataAnalysisChallenge/city_search.json') as f:
    data = json.load(f)

# Getting data to a list, each element of this list is a string of the searched cities by one client
list_of_queries = [data[i]['cities'][0] for i in range(len(data))]

# # l is a list of lists of words of the queries
l = [list_of_queries[i].split(', ') for i in range(len(list_of_queries))]

# We prepare 4 training sets for our 4 fold cross validation
all_sentences = list()
all_sentences.append(l[0:math.floor(0.75*len(l))])
all_sentences.append(l[0:math.floor(0.5*len(l))] + l[math.floor(0.75*len(l)):])
all_sentences.append(l[0:math.floor(0.25*len(l))] + l[math.floor(0.5*len(l)):])
all_sentences.append(l[math.floor(0.25*len(l)):])

#We prepare 4 testing set for 4 fold cross validation
sentences_remaining= list()
for i in range(4):
    sentences_remaining.append(l[math.floor((100-((i+1)*0.25)*len(l))):math.floor((100-(0.25*i))*len(l))])

results = list()
average = 0

#Loop for the 4-fold Cross Validation
for k in range(4):

    training_set = all_sentences[k]

    # Here we create our model, and we train it with a training set called training_set
    # we use 200 epoch after trying 5 to 500
    model = gensim.models.Word2Vec(training_set, min_count=1)
    model.train(training_set, total_examples=model.corpus_count, epochs=200)


    answers = list()
    # Now we build our testing set, by taking the remaining sentences
    test = sentences_remaining[k]
    testing_set = list()

    # Each element of test is a list of cities. For each of those lists we
    #  take the last city as answer and the rest as a testing query.
    # The goal is to find the good city given a query
    for i in range(len(test)):
        if len(test[i]) > 3 :
            answers.append(test[i][len(test[i])-1])
            testing_set.append(test[i][0:len(test[i])-1])

    #Here we test the trained model with our testing set and gather the answers.
    #After that we compare the answers with the true ones.

    answers2 = list()
    for i in range(len(testing_set)):
        #print('test set',i,testing_set[i])
        l2 = list()
        for j in range(5):
            l2.append(model.most_similar(testing_set[i])[j][0])
        answers2.append(l2)

    # # Compute accuracy
    results.append(sum(1 for x,y in zip(answers,answers2) if x in y) / len(answers))
    average = average + sum(1 for x,y in zip(answers,answers2) if x in y) / len(answers)
print(average/4.0)
print(results)










#Useless in this case, but theoritically, we should make sure that the training
# set contains all the vocabulary that the testing set use to avoid error : "word 'XYZ' not in vocabulary"


# # # Get the list of all the cities
# liste_des_villes = list()
# for i in range(len(l)):
#     for j in range(len(l[i])):
#         liste_des_villes.append(l[i][j])
#
# liste_des_villes = list(set(liste_des_villes))
#
# sentences = list_of_queries
# training_set = sentences[0:math.floor(0.75*len(sentences))]
# # Training set building to get all the vocab
# minilist = list()
# for i in liste_des_villes:
#     for j in range(len(sentences)):
#         if i in sentences[j]:
#             minilist.append(sentences[j])
# #           print(i)
#             break
# training_set = training_set + minilist
