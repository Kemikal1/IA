KnnClassifier
 -cele doua metode au ca argument
 -test_data - datele care trb clasificate 
 -metric - tipul distantei folosite de clasificator(l1 sau l2)
 -num_neighbors - numarul de vecini cei mia apropiati pe care se face compararea
 
 -clasifiy_data(test_data,num_neigbors,metric)
  -clasifica o fraza 
 -clasify_datas(test_data,num_neigbors,metric)
  -clasifica un vector de fraze


get_id(date)
 -returneaza idurile din portiunea de date din argument
get_id_label(date)
 -returneaza id-urile etichetelor

get_data(date)
 -returneaza datele cu fiecare fraza impartita in cuvinte
remove_id(date)
 -returneaza datele din perechea id--fraza

accuaracy(true,pred)
 -returneaza accuaratetea dintre doui vectori de etichete
get_labels()
 -scoate din perechea id--eticheta etichetele

normalize_data(train_data,test_data,type=None)
 -normalizeaza doua perechi de date ,luand ca al 3-lea argument 
 -tipul de normalizare dorit
 -l1 - normalizarea l1
 -l2 - normalizarea l2
 -min-max - normalizarea min-max
 -standard - normalizare standar
