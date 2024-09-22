# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:30:19 2020

@author: ludov
"""

"""""""""""""""""""""""""""""""""""""""
création de plusieurs fonctions:
-récupération et analyse de données
-exportation au format CSV
"""""""""""""""""""""""""""""""""""""""

#import time

#Debut du decompte du temps
#start_time = time.time()



def data_collection():
    # importation des librairies
    from bs4 import BeautifulSoup
    import requests
    
    
    
    """""""""""""""""""""""""""""""""""""""""""""
    Étape 1 : Envoi d’une requête HTTP à une URL
    """""""""""""""""""""""""""""""""""""""""""""
    
    url = "https://www.worldometers.info/coronavirus/"
    # réalisation d'une requête GET pour récupérer le contenu HTML brut
    html_content = requests.get(url).text
    
    """""""""""""""""""""""""""""""""
    Etape 2 : Analyse du contenu html
    """""""""""""""""""""""""""""""""
    
    soup = BeautifulSoup(html_content, "lxml")
    #impression des données analysées de html : print(soup.prettify()) 
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Etape 3 : Analyse  de la balise HTML, où se trouve le contenu
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    #récupération du tableau des données relatives au COVID
    COVID_table = soup.find("table", attrs={"class": "table"}) 
    
    # récupération des éléments ligne par ligne
    COVID_table_data = COVID_table.tbody.find_all("tr")  
    
    """""""""""""""""""""""""""""""""
    Etape 4 : Récupération des titres
    """""""""""""""""""""""""""""""""
    
    t_headers = []
    for th in COVID_table.find_all("th"):
       #suppression des nouvelles lignes et des espaces supplémentaires à gauche et à droite
       t_headers.append(th.text.replace('\n', ' ').strip())
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Etape 5 : récupération de toutes les données dans une liste
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    data_body = []
    for n in range(0,len(COVID_table_data)):
        #récupération de chaque élément du tableau
        for td in COVID_table_data[n].find_all("td"):
            #suppression des nouvelles lignes et des espaces supplémentaires à gauche et à droite
            data_body.append(td.text.replace('\n', ' ').strip())
            
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Etape 6 : création d'un dictionnaire itéré afin de classer les données colonne par colonne
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    data={}
    data_export={}
    #création d'un dictionnaire pour chaque colonne et exportation dans un dictionnaire final
    for n in range(0,len(t_headers)):
        data={str(t_headers[n]):data_body[n::len(t_headers)]}
        data_export.update(data)
    
    """""""""""""""""""""""""""""""""  
    Etape 7 : traitement de données
    """""""""""""""""""""""""""""""""
    
    #liste de pays
    pays=(data_export[t_headers[0]])
    #selection d'un pays
    user_input=input("choose a country to learn more about its data" +('\n'*2))
    for a in range(0,len(COVID_table_data)):
        if user_input==pays[a]:
            i=1
            for r in list(t_headers[1:len(t_headers)]):
                print(t_headers[i])
                i=i+1
                print(data_export[r][a])
                if data_export[r][a]=='':
                    print(0)
            print('\n\n')
    
       
    #liste des cas totaux
    # conversion de la liste string en int
    total_cases=(data_export[t_headers[1]]) 
    total_cases= list(map(lambda b: b.replace(",",""), total_cases))
    total_cases = list(map(float, total_cases))
    total_cases, pays =zip(*sorted(zip(total_cases, pays),reverse=True))
    list(total_cases)
    list(pays)
    #suppression  des avertissements matplotlib
    import matplotlib.pyplot as plt
    import warnings
    import matplotlib.cbook
    warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
    #création d'un histogramme des cas totaux
    for n in range(0,10):
        plt.rcParams["figure.figsize"] = (20,25)
        histo1=plt.subplot(3,1,1)
        #n+1 : exclure World
        histo1.bar(pays[n+1],total_cases[n+1])
        plt.title("classement par totalité de cas")
    plt.show(histo1)
        
        
    #liste des cas actifs
    #reconversion de la liste des lieux dans leur disposition intiale
    pays=(data_export[t_headers[0]])
    # conversion de la liste string en int
    active_cases=(data_export[t_headers[6]]) 
    active_cases= list(map(lambda b: b.replace(",",""), active_cases))
    active_cases = list(map(float,active_cases))
    active_cases, pays =zip(*sorted(zip(active_cases, pays),reverse=True))
    list(active_cases)
    list(pays)
    #création d'un histogramme des cas encore actifs
    for n in range(0,10):
        histo2=plt.subplot(3,1,2)
        #n+1 : exclure World
        histo2.bar(pays[n+1],active_cases[n+1]) 
        plt.title("classement par cas actifs") 
        plt.rcParams["figure.figsize"] = (20,25)
    plt.show(histo2)
    print('\n'*2)
    
        
    #liste des nouveaux cas
    # conversion de la liste string en int
    new_cases_str=data_export[t_headers[2]] 
    new_cases_str= list(map(lambda b: b.replace(",",""), new_cases_str))
    new_cases = []
    for i in new_cases_str :
        if i != '':
            new_cases.append(int(i))
        else:
            new_cases.append(0)
    #boucle excluant les continents (n commence à 7)
    for n in range(7,len(COVID_table_data)-1):
        #n+1 : exclure World
        if new_cases[n+1]>1000:
            print("The virus is spreading quiclky in" +' '+ data_export[t_headers[0]][n+1])
    print("\n"*1)
    
    
    #liste des nouvelles morts
    # conversion de la list string en int
    new_deaths_str=data_export[t_headers[4]] 
    new_deaths_str= list(map(lambda b: b.replace(",",""), new_deaths_str))
    new_deaths = []
    for i in new_deaths_str :
        if i != '':
            new_deaths.append(int(i))
        else:
            new_deaths.append(0)
    #boucle excluant les continents (n commence à 7)
    for n in range(7,len(COVID_table_data)-1):
        #n+1 : exclure World
        if new_deaths[n+1]>200:
            print("The virus is becoming strongly lethal in" +' '+ data_export[t_headers[0]][n+1])
    print('\n'*2)
    return(data_export)
    


 #Exportation au format CSV du tableau
def export_csv(): 
    
    # importation des librairies
    import pandas as pd 
    #création du tableau
    df=pd.DataFrame(data_collection())
    df.to_csv(r'data_covid.csv', index = False)
    print(df)



"""""""""""""""""""""""
execution des fonctions
"""""""""""""""""""""""
export_csv()


#Affichage du temps d execution
#print("Temps d execution : %s secondes ---" % (time.time() - start_time))