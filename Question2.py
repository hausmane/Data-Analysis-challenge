import gensim
import json

# Read the json file
with open('/home/user/Téléchargements/DataAnalysisChallenge/city_search.json') as f:
    data = json.load(f)


# Getting data to a list
list_of_queries = [data[i]['cities'][0] for i in range(len(data))]
#
# l is a list of lists of words of the queries
short_list = list(set(list_of_queries))
l = [short_list[i].split(', ') for i in range(len(short_list))]

# # Get the list of all the cities in the database
liste_des_villes = list()
for i in range(len(l)):
    for j in range(len(l[i])):
        liste_des_villes.append(l[i][j])

liste_des_villes = list(set(liste_des_villes))
print(liste_des_villes)


sentences = l

# # Here we create our moodel to predict the city
model = gensim.models.Word2Vec(sentences, min_count=1)
model.train(sentences, total_examples=model.corpus_count, epochs=200)

#
# #Previous_city is ncity list, it returns the most likely next city to be searched
def next_city(previous_cities):
    if is_good_list_city(previous_cities,liste_des_villes):
        str = '!!'.join(previous_cities)
        return model.most_similar(str.split('!!'), topn=3)
    else:
        print('Wrong list')
#
# This function verify if the given city list is ok, to avoid error like "word 'XYZ' not in vocabulary"
def is_good_list_city(list, liste_des_villes):
    for city in list:
        if city in liste_des_villes:
            print(city, ' : Correct city')
        else:
            print(city, ' : Wrong city')
            return False
    return True


print(next_city(['Montreal QC', "Quebec QC"]))

