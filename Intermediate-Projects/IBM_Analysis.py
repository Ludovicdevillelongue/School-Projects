"""""""""""""""
 Exercice 1
"""""""""""""""
# Ci-dessous l'integralite des packages necessaires (d'apres ma correction)  

from pandas import *
from scipy.stats import chi2
import numpy as np
import numpy.random as npr
import datetime as dt
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.optimize import minimize
import pandas as pd
from sklearn.decomposition import PCA
import warnings

###############################################################################
#                            Application theorique                            #
###############################################################################  

# Soit T le nombre d'observations a simuler
T= npr.randint(2, size=1000)
# Soit p le paramètre de la loi de Bernoulli, cad la probabilite associee a la VaR
p = 0.05
# Simuler la série I (T,1) contenant des 0 et des 1, les 1 apparaissent avec une probabilte p
I=npr.binomial(T, p)
# Compter le nombre d'exceptions (violations, breach or hit, = 1) de la serie I, note n_1
n_1=np.count_nonzero(I == 1)
# Compter le nombre d'absence d'exceptions (= 0) de la serie I, note n_0
n_0= np.count_nonzero(I == 0)

#Calcul de pi chapeau, de L(pi) et de L(p), de la statistique de test et de la p.value       

" Vraisemblance sous l'hypothese nulle (H0) pour le test de couverture non conditionnelle "
# On note cette vraisemblance L_p, elle est donnee par l'equation (1)
L_p=((1-p)**(n_0))*(p**(n_1))
" Vraisemblance sous l'hypothese alternative (H1) pour le test de couverture non conditionnelle "
# Soit pi_hat l'estimateur du maximum de vraisemblance de pi
pi_hat=n_1/(n_0+n_1)
# On note cette vraisemblance L_pi, elle est donnee par l'equation (2)
L_pi=((1-pi_hat)**(n_0))*(pi_hat**(n_1))
" Ratio de Vraisemblance pour le test de couverture non conditionnelle "
" Likelihood Ratio: -2*(LogLikeH0)-LogLikeH1)) "
# On note ce ration LR_uc, il est donne par l'equation (3)
LR_uc=-2*np.log(L_p/L_pi)
" Valeur critique - Quantile d'une distribution de Khi-deux à 1 ddl "
# On note ce quantile a 1-p (1-seuil de risque) QuantChi2
QuantChi2=chi2.cdf(1-p,df=1)
# Condition if-then permettant de vérifier si l'on rejette HO ou non (rejet si LR_uc > QuantChi2) => 4 lignes
if LR_uc>QuantChi2:
    print("\nApplication Théorique EX1:\n\
réponse EX1 question 5 :\
Rejet de l'hypothese nulle du test de couverture non conditionnelle",'\n')
else:
    print('\n\n',"Non Rejet de l'hypothese nulle du test de couverture non conditionnelle",'\n')

# Question bonus :
# Condition if-then permettant de vérifier si l'on rejette HO ou non (rejet si Pvalue(LR_uc) < p)  => 4 lignes
# Pvalue (chi2.sf(LR_uc, 1) = 1-chi2.cdf(LR_uc , 1))"
if (1-chi2.cdf(LR_uc , 1))<p:
    print("Rejet de l'hypothese nulle du test de couverture non conditionnelle par la p-value",'\n\n\n\n\n\n')
else:
    print("Non Rejet de l'hypothese nulle du test de couverture non conditionnelle par la p-value",'\n\n\n\n\n\n')

###############################################################################
#                            Application empirique                            #
###############################################################################  

# Charger les donnes du classeur Excel "Data4Exam_Ex1.xlsx" dans IBM (type DataFrame)
IBM=ExcelFile(r"Data4Exam_Ex1.xlsx")
# Dans ibm_cours (type DataFrame), récupérer les données de l'onglet 'IBM' en utilisant l'instruction parse
ibm_cours=IBM.parse('IBM')
# Ne conserver dans ibm que la variable cours du DataFrame ibm_cours
ibm=ibm_cours['Cours']
# Calculer dans ibm_rend les rendements logarithmiques de ce cours boursiers (sans faire de boucle)
ibm_rend=np.log(ibm) - np.log(ibm.shift(1))
# Selectionner dans ibm_rend les rendements allant de l'index 1 a T, afin de ne pas conserver la premiere observation qui est NaN
ibm_rend=ibm_rend.iloc[1:len(ibm_rend)]
# Soit T le nombre total d'observation de ibm_rend 
T=len(ibm_rend)
# Selectionner le seuil de risque alpha (ici alpha = 1%)
alpha=0.01
# Trier les rendements de ibm_rend par ordre croissant dans ibm_rend_sort
ibm_rend_sort=ibm_rend.sort_values(ascending=True)
# Recuperer dans VaR le alpha ième percentile (cad la alpha*T ième observation de ibm_rend_sort)
VaR=ibm_rend_sort.take([alpha*T])
# Dupliquer ce scalaire afin d'obtenir un vecteur que l'on nomme de nouveau VaR mais de dimension (T,1)
VaR= np.repeat(VaR,T)

# Graphique Figure 1 
# Recuperer dans Date la variable Date du DataFrame ibm_cours
Date=ibm_cours['Date']
# Redimensionner ce vecteur de Date
Date=Date[1:len(Date)]
# Debut instruction fig => 8 lignes
plt.rcParams["figure.figsize"] = (30,12)
plt.figure(1)
plt.plot(Date, ibm_rend, label= 'Daily Returns')
plt.plot(Date, VaR, label='Historical VaR')
plt.legend(loc='upper right', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.ylabel('IBM Daily Returns with its historical VaR', fontsize=12)
plt.title('IBM Returns with its historical VaR at 1%', fontsize=16)
plt.show()






#############################

# Construire la serie d'exceptions que l'on nomme I (une exception a lieu lorsque rendement est inferieur a la VaR)
I= Series(np.where(ibm_rend<VaR.iloc[1], 1, 0), name="Cours") 
#I.loc[ibm_rend<VaR.iloc[0], 0] = 1
# Compter le nombre d'exceptions (violations, breach or hit, = 1) de la serie I, note n_1
n_1=np.count_nonzero(I == 1)
# Compter le nombre d'absence d'exceptions (= 0) de la serie I, note n_0
n_0=np.count_nonzero(I == 0)
# Compter le nombre total d'observation de I, note deno (verifier que n_0 + n_1 = T)
deno=len(I)
print('Application Empirique EX1\n'\
"l'égalité n_0 + n_1 = T est "+str(n_0+n_1==T),'\n\n\n')


"""  unconditional coverage test """

" Vraisemblance sous l'hypothese nulle (H0) pour le test de couverture non conditionnelle "
# On note cette vraisemblance L_uc_H0, elle est donnee par l'equation (1) où p=alpha ici
L_uc_H0=((1-alpha)**n_0)*(alpha**n_1)
" Vraisemblance sous l'hypothese alternative (H1) pour le test de couverture non conditionnelle "
# Soit pi_hat_uc l'estimateur du maximum de vraisemblance de pi
pi_hat_uc=n_1/(n_0+n_1)
# On note cette vraisemblance L_pi, elle est donnee par l'equation (2)
L_pi=((1-pi_hat_uc)**(n_0))*(pi_hat_uc**(n_1))
" Ratio de Vraisemblance pour le test de couverture non conditionnelle "
# On note ce ration LR_uc, il est donne par l'equation (3)
LR_uc=-2*np.log(L_uc_H0/L_pi)
# Calcul de Pvalue_uc, la Pvalue associe a cette statistique (1-cdf d'un Khi-deux a 1 ddl pris en LR_uc)
Pvalue_uc=1-chi2.cdf(LR_uc,df=1)
"""  independence test """
# Dans I_lead recuperer les observations de I allant de l'index 1 a T
I_lead=I[1:T]
# Dans I_lag recuperer les observations de I allant de l'index 0 a T-1
I_lag=I[0:T-1]
# Initialisation de n_11 à 0
n_11=0
# Initialisation de n_10 à 0
n_10=0
# Initialisation de n_00 à 0
n_00=0
# Initialisation de n_01 à 0
n_01=0
# Boucle for allant de l'index 0 à T-1 permettant d'incrementer les compteurs des sequences (n_11, n_10, n_00 et n_01)
for i in range(len(I_lag)-1):
    # Condition if permmettant d'incrémenter n_11 => 2 lignes
    if (I_lag[i]==1 and I_lag[i+1]==1):
        n_11+=1

    # Condition if permmettant d'incrémenter n_10 => 2 lignes
    if (I_lag[i]==1 and I_lag[i+1]==0):
        n_10+=1
    # Condition if permmettant d'incrémenter n_00 => 2 lignes
    if (I_lag[i]==0 and I_lag[i+1]==0):
        n_00+=1
    # Condition if permmettant d'incrémenter n_01 => 2 lignes
    if (I_lag[i]==0 and I_lag[i+1]==1):
        n_01+=1
# Verifier que le nombre d'observation de I_lead est bien egal a la somme suivante n_11 + n_10 + n_00 + n_01
I_lead==n_11 + n_10 + n_00 + n_01
" Vraisemblance sous l'hypothese alternative (H0) pour le test d'independance "
# Soit pi_hat_ind l'estimateur du maximum de vraisemblance de pi_ind
pi_hat_ind= (n_01+n_11)/(n_11+n_10+n_00+n_01) 
# On note cette vraisemblance L_ind_H0, elle est donnee par l'equation (4)
L_ind_H0=((1-pi_hat_ind)**(n_00+n_10))*(pi_hat_ind**(n_01+n_11))
" Vraisemblance sous l'hypothese alternative (H1) pour le test d'independance "
# Soit pi_hat_01 l'estimateur du maximum de vraisemblance de pi_01
pi_hat_01= n_01/(n_00+n_01)
# Soit pi_hat_11 l'estimateur du maximum de vraisemblance de pi_11
pi_hat_11=n_11/(n_10+n_11)
# On note cette vraisemblance L_ind_H1, elle est donnee par l'equation (5)
L_ind_H1=((1-pi_hat_01)**n_00)*(pi_hat_01**(n_01))*((1-pi_hat_11)**n_10)*(pi_hat_11**n_11)
" Ratio de Vraisemblance pour le test d'independance  "
# On note ce ratio LR_ind, il est donne par l'equation (6)
LR_ind=-2*np.log(L_ind_H0/L_ind_H1)
# Calcul de Pvalue_ind, la Pvalue associe a cette statistique (1-cdf d'un Khi-deux a 1 ddl pris en LR_ind)
Pvalue_ind=1-chi2.cdf(LR_ind,df=1)

""" conditional coverage test """
" Ratio de Vraisemblance pour le test de couverture conditionnelle  "
# On note ce ratio LR_cc, il est donne par l'equation (7)
LR_cc=-2*np.log(L_uc_H0/L_ind_H1)
# Calcul de Pvalue_cc, la Pvalue associe a cette statistique (1-cdf d'un Khi-deux a 2 ddl pris en LR_cc)

Pvalue_cc=1-chi2.cdf(LR_cc,df=2)

# Question bonus : 
# To get the equality,  LR_cc = LR_uc + LR_ind, 
# il faut remplacer L_uc_H0 par L_ind_H0 dans le calcul de la stat LR_uc (p8 du pdf)
LR_uc_check = -2*np.log(L_uc_H0/L_ind_H0)
LR_cc_check = LR_uc_check + LR_ind

print("réponse Ex1 question 6:\n\
      Le quantile relatif à un seuil de risque α=1% pour une loi de khi deux à 1 degré de liberté a pour valeur 6.63. \n"
"La statistique de test LRuc étant égal à 0.001, elle est donc inférieure au quantile observé, ce qui conduit \
à ne pas rejeter H0 au seuil de risque α=1%. La VaR(p=α) est donc une bonne mesure de risque pour cette série \
de rendements puisque le nombre d’exception est proche de 1%.\n "
"La statistique de test LRind étant égal à 10.081, elle est donc supérieure au quantile observé, \
ce qui conduit à rejeter H0 au seuil de risque α=1%. Formellement, la variable It(α) associée à une \
exception de la VaR en t pour un taux de couverture α est indépendante de la variable It-k(α) pour tout k différent de 0. \
Sachant que le test est rejeté, cela signifie que les exceptions passées de la VaR détiennent plus de 1% des informations \
sur les exceptions présentes et futures. \n"
"     Le quantile relatif à un seuil de risque α=1% pour une loi de khi deux à 2 degrés de liberté a pour valeur 9.21.\n" 
"La statistique de test LRcc étant égal à 10.123, elle est donc supérieure au quantile observé, \
ce qui conduit à rejeter H0 au seuil de risque α=1%. Le test de couverture conditionnel combine \
les deux tests précédents, d’où l’égalité LRcc=LRind+LRuc. Sachant que le test est rejeté, cela \
signifie que les exceptions passées de la VaR détiennent plus de 1% des informations sur les exceptions \
présentes et futures et que la VaR(p=α) ne serait pas une bonne mesure de risque pour la série de rendements.\n"
"     Une p-value petite signifie qu’il y a davantage de preuves en faveur de l’hypothèse alternative, \
ce qui est bien le cas pour LRind et LRcc mais n’est pas vrai pour LRuc. Les p-values viennent donc \
confirmer les résultats précédents.",'\n\n\n')




""" BIS traffic light """
# Dans year, ne conserver que l'annee de la variable Date du DataFrame ibm_cours
year=ibm_cours['Date'].dt.year
# Dans breach, joindre le vecteur year au vecteur I en se basant sur les index de I, breach est donc de dimension (T,2)
breach=concat((year,I), axis=1)
# Renommer les deux variables de breach, 'Cours' par 'Hit' et 'Date' par 'Year'
breach.rename(columns={"Cours": "hit", "Date": "Year"})
# Creer breach_peryear contenant le nombre d'exceptions par année (dimension (10,1))
breach_peryear=breach.groupby(['Date']).apply(lambda x: x[x == 1].count().drop(['Date']))
print('réponse Ex1 question 7: ')
# Boucle for sur le nombre d'observation de breach_peryear
for i in range(len(breach_peryear)):
    # Recuperer dans hit la ieme observation de breach_peryear (cad le nombre d'exceptions pour la ieme annee)
    hit=breach_peryear.iloc[i,0]
    # Condition if-elif-else permettant de determiner la zone (rouge, jaune ou vert) => 6 lignes
    if hit<5:
        zone="verte"
    elif hit>=5 and hit<10:
        zone="jaune"
    else:
        zone="rouge"
        
    # reporter le resultat par une instruction print pour l'annee que vous etudiez
    print("la zone pour l'année "+ str(breach_peryear.index[i])+ " est "+str(zone))
print('\n\n\n')

    
""" AR(1)-GJR-GARCH(1,1) """
# Definir un modele AR(1)-GJR-GARCH(1,1) (Equations 1-3) avec des innovations suivant une loi de Student
# sur les rendements d'IBM multiplie par 100, sauvegarder ce modele dans MyModel


MyModel = arch_model(ibm_rend*100, vol='Garch', p=1, o=1, q=1, dist='StudentsT')

# Estimer le modele MyModel, sauvegarder le resultat dans MyResult
MyResult=MyModel.fit()


# Reporter dans la console 1/A le tableau recapitulant cette regression
print('\n\n',MyResult.summary(),'\n\n\n')
# Sauvegarder dans MyVol la volatilite conditionnelle de ce modele
MyVol=MyResult.conditional_volatility
# Le modele etant autoregressif, nous souhaitons enlever la premiere observation car elle est NaN
#non nécessaire de redimensionner
year=year[1:len(year)]

# Grapher cette volatilite conditionnelle
plt.figure(2)
plt.plot(Date, MyVol)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Conditional Volatility', fontsize=12)
plt.title('Conditional volatility through time', fontsize=16)
plt.show()
# Caculer dans VaR_garch la nouvelle mesure de risque associée à ce modèle (à vous de trouver la formule pour le calcul)
q=MyModel.distribution.ppf([alpha], MyResult.params[-1:])
mu=MyResult.params.mu
VaR_garch=(mu+MyVol*q)/100

# grapher la VaR
plt.figure(3)
plt.rcParams["figure.figsize"] = (30,12)
plt.plot(Date, ibm_rend, label= 'Daily Returns')
plt.plot(Date, VaR_garch, label='VaR_garch')
plt.legend(loc='upper right', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.ylabel('IBM Daily Returns with its VaR_garch', fontsize=12)
plt.title('IBM Returns with its VaR_Garch at 1%', fontsize=16)
plt.show()








#########################################reprise des données avec VaR_garch######################################

# Construire la serie d'exceptions que l'on nomme I (une exception a lieu lorsque rendement est inferieur a la VaR)

I_garch=Series(np.where(ibm_rend[:]<VaR_garch[:],1,0),name="Cours")


# Compter le nombre d'exceptions (violations, breach or hit, = 1) de la serie I, note n_1
n_1_garch=np.count_nonzero(I_garch == 1)
# Compter le nombre d'absence d'exceptions (= 0) de la serie I, note n_0
n_0_garch=np.count_nonzero(I_garch == 0)
# Compter le nombre total d'observation de I, note deno (verifier que n_0 + n_1 = T)
deno_garch=len(I_garch)



"""  unconditional coverage test """

" Vraisemblance sous l'hypothese nulle (H0) pour le test de couverture non conditionnelle "
# On note cette vraisemblance L_uc_H0, elle est donnee par l'equation (1) où p=alpha ici
L_uc_H0_garch=((1-alpha)**(n_0_garch))*(alpha**(n_1_garch))
" Vraisemblance sous l'hypothese alternative (H1) pour le test de couverture non conditionnelle "
# Soit pi_hat_uc l'estimateur du maximum de vraisemblance de pi
pi_hat_uc_garch=n_1_garch/(n_0_garch+n_1_garch)
# On note cette vraisemblance L_pi, elle est donnee par l'equation (2)
L_pi_garch=((1-pi_hat_uc_garch)**(n_0_garch))*(pi_hat_uc_garch**(n_1_garch))
" Ratio de Vraisemblance pour le test de couverture non conditionnelle "
# On note ce ration LR_uc, il est donne par l'equation (3)
LR_uc_garch=-2*np.log(L_uc_H0_garch/L_pi_garch)
# Calcul de Pvalue_uc, la Pvalue associe a cette statistique (1-cdf d'un Khi-deux a 1 ddl pris en LR_uc)
Pvalue_uc_garch=1-chi2.cdf(LR_uc_garch,df=1)
"""  independence test """
# Dans I_lead recuperer les observations de I allant de l'index 1 a T
I_lead_garch=I_garch[1:T]
# Dans I_lag recuperer les observations de I allant de l'index 0 a T-1
I_lag_garch=I_garch[0:T-1]
# Initialisation de n_11 à 0
n_11_garch=0
# Initialisation de n_10 à 0
n_10_garch=0
# Initialisation de n_00 à 0
n_00_garch=0
# Initialisation de n_01 à 0
n_01_garch=0
# Boucle for allant de l'index 0 à T-1 permettant d'incrementer les compteurs des sequences (n_11, n_10, n_00 et n_01)
for i in range(len(I_lag_garch)-1):
    # Condition if permmettant d'incrémenter n_11 => 2 lignes
    if (I_lag_garch[i]==1 and I_lag_garch[i+1]==1):
        n_11_garch+=1

    # Condition if permmettant d'incrémenter n_10 => 2 lignes
    if (I_lag_garch[i]==1 and I_lag_garch[i+1]==0):
        n_10_garch+=1
    # Condition if permmettant d'incrémenter n_00 => 2 lignes
    if (I_lag_garch[i]==0 and I_lag_garch[i+1]==0):
        n_00_garch+=1
    # Condition if permmettant d'incrémenter n_01 => 2 lignes
    if (I_lag_garch[i]==0 and I_lag_garch[i+1]==1):
        n_01_garch+=1
# Verifier que le nombre d'observation de I_lead est bien egal a la somme suivante n_11 + n_10 + n_00 + n_01
I_lead_garch==n_11_garch + n_10_garch + n_00_garch + n_01_garch
" Vraisemblance sous l'hypothese alternative (H0) pour le test d'independance "
# Soit pi_hat_ind l'estimateur du maximum de vraisemblance de pi_ind
pi_hat_ind_garch= (n_01_garch+n_11_garch)/(n_11_garch+n_10_garch+n_00_garch+n_01_garch) 
# On note cette vraisemblance L_ind_H0, elle est donnee par l'equation (4)
L_ind_H0_garch=((1-pi_hat_ind_garch)**(n_00_garch+n_10_garch))*(pi_hat_ind_garch**(n_01_garch+n_11_garch))
" Vraisemblance sous l'hypothese alternative (H1) pour le test d'independance "
# Soit pi_hat_01 l'estimateur du maximum de vraisemblance de pi_01
pi_hat_01_garch= n_01_garch/(n_00_garch+n_01_garch)
# Soit pi_hat_11 l'estimateur du maximum de vraisemblance de pi_11
pi_hat_11_garch=n_11_garch/(n_10_garch+n_11_garch)
# On note cette vraisemblance L_ind_H1, elle est donnee par l'equation (5)
L_ind_H1_garch=((1-pi_hat_01_garch)**n_00_garch)*(pi_hat_01_garch**(n_01_garch))*((1-pi_hat_11_garch)**n_10_garch)*(pi_hat_11_garch**n_11_garch)
" Ratio de Vraisemblance pour le test d'independance  "
# On note ce ratio LR_ind, il est donne par l'equation (6)
LR_ind_garch=-2*np.log(L_ind_H0_garch/L_ind_H1_garch)
# Calcul de Pvalue_ind, la Pvalue associe a cette statistique (1-cdf d'un Khi-deux a 1 ddl pris en LR_ind)
Pvalue_ind_garch=1-chi2.cdf(LR_ind_garch,df=1)

""" conditional coverage test """
" Ratio de Vraisemblance pour le test de couverture conditionnelle  "
# On note ce ratio LR_cc, il est donne par l'equation (7)
LR_cc_garch=-2*np.log(L_uc_H0_garch/L_ind_H1_garch)
# Calcul de Pvalue_cc, la Pvalue associe a cette statistique (1-cdf d'un Khi-deux a 2 ddl pris en LR_cc)

Pvalue_cc_garch=1-chi2.cdf(LR_cc_garch,df=2)

# Question bonus : 
# To get the equality,  LR_cc = LR_uc + LR_ind, 
# il faut remplacer L_uc_H0 par L_ind_H0 dans le calcul de la stat LR_uc (p8 du pdf)
LR_uc_check_garch = -2*np.log(L_uc_H0_garch/L_ind_H0_garch)
LR_cc_check_garch = LR_uc_check_garch + LR_ind_garch

print("réponse Ex1 question 8 partie 1:\n\
      Le quantile relatif à un seuil de risque α=1% pour une loi de khi deux à 1 degré de liberté a pour valeur 6.63. \n"
"La statistique de test LRuc garch étant égal à 0.882, elle est donc inférieure au quantile observé, ce qui conduit \
à ne pas rejeter H0 au seuil de risque α=1%. La VaR(p=α) est donc une bonne mesure de risque pour cette série \
de rendements puisque le nombre d’exception est proche de 1%.\n "
"La statistique de test LRind garch étant égal à 0.724, elle est donc inférieure au quantile observé, \
ce qui conduit à ne pas rejeter H0 au seuil de risque α=1%. Formellement, la variable It(α) associé à une \
exception de la VaR en t pour un taux de couverture α est indépendante de la variable It-k(α) pour tout k différent de 0. \
Sachant que le test n'est pas rejeté, cela signifie que les exceptions passées de la VaR ne détiennent pas plus de 1% des informations \
sur les exceptions présentes et futures. \n"
"     Le quantile relatif un seuil de risque α=1% pour une loi de khi deux à 2 degrés de liberté a pour valeur 9.21.\n" 
"La statistique de test LRcc garch étant égal à 1.654, elle est donc inférieure au quantile observé, \
ce qui conduit à ne pas rejeter H0 au seuil de risque α=1%. Le test de couverture conditionnel combine \
les deux tests précédents, d’où l’égalité LRcc=LRind+LRuc. Sachant que le test n'est pas rejeté, cela \
signifie que les exceptions passées de la VaR ne détiennent pas plus de 1% des informations sur les exceptions \
présentes et futures et que la VaR(p=α) serait donc une bonne mesure de risque pour la série de rendements.",'\n\n\n')



""" BIS traffic light """
# Dans year, ne conserver que l'annee de la variable Date du DataFrame ibm_cours
year=ibm_cours['Date'].dt.year
# Dans breach, joindre le vecteur year au vecteur I en se basant sur les index de I, breach est donc de dimension (T,2)
breach_garch=concat((year,I_garch), axis=1)
# Renommer les deux variables de breach, 'Cours' par 'Hit' et 'Date' par 'Year'
breach_garch.rename(columns={"Cours": "hit", "Date": "Year"})
# Creer breach_peryear contenant le nombre d'exceptions par année (dimension (10,1))
breach_peryear_garch=breach_garch.groupby(['Date']).apply(lambda x: x[x == 1].count().drop(['Date']))
print('réponse Ex1 question 8 partie 2: ')
# Boucle for sur le nombre d'observation de breach_peryear
for i in range(len(breach_peryear_garch)):
    # Recuperer dans hit la ieme observation de breach_peryear (cad le nombre d'exceptions pour la ieme annee)
    hit_garch=breach_peryear_garch.iloc[i,0]
    # Condition if-elif-else permettant de determiner la zone (rouge, jaune ou vert) => 6 lignes
    if hit_garch<5:
        zone_garch="verte"
    elif hit_garch>=5 and hit_garch<10:
        zone_garch="jaune"
    else:
        zone_garch="rouge"
        
    # reporter le resultat par une instruction print pour l'annee que vous etudiez
    print("la zone pour l'année "+ str(breach_peryear_garch.index[i])+ " est "+str(zone_garch))
print('\n\n\n')












"""""""""""""""
 Exercice 2
"""""""""""""""

# Creer la fonction _portfolio_risk(weights, covariances)
def _portfolio_risk(weights, covariances):
    # Calculer portfolio_risk, le risque (la volatilite) du portefeuille d'apres l'Equation (7) en utilisant les weights et la matrices de covariances
    portfolio_risk = np.sqrt(weights * covariances * weights.T)
    # La fonction _portfolio_risk(weights, covariances) renvoie la volatilite du portefeuille (nommee portfolio_risk)
    return portfolio_risk

# Creer la fonction _assets_risk_contribution_to_portfolio_risk(weights, covariances)
def _assets_risk_contribution_to_portfolio_risk(weights, covariances):
    # Calculer portfolio_risk, le risque du portefeuille en utilisant la fonction _portfolio_risk(weights, covariances)
     portfolio_risk = _portfolio_risk(weights, covariances)
    # Calculer assets_risk_contribution de dimension (nb_assets,1), la contribution de chaque actif au risque du portefeuille en utilisant l'Equation (8) 
     assets_risk_contribution = np.multiply(weights.T, covariances * weights.T)  / portfolio_risk
       
    # La fonction _assets_risk_contribution_to_portfolio_risk(weights, covariances) renvoie la contribution de chaque actif au risque du portefeuille (nommee assets_risk_contribution)
     return assets_risk_contribution

# Creer la fonction _risk_budget_objective_error(weights, args)
def _risk_budget_objective_error(weights, args):
    # Recuperer la matrice de covariance qui occupe la premiere position dans la variable args 
    covariances = args[0]
    # Recuperer la contribution desiree de chaque actif au risque du portefeuille, assets_risk_budget occupe la seconde position dans la variable args 
    assets_risk_budget = args[1]
    # Convertir la variable weights de type array en une matrice nommee egalement weights (indice: utiliser la fonction np.matrix())
    weights = np.matrix(weights)
    # Calculer portfolio_risk, le risque du portefeuille en utilisant la fonction _portfolio_risk(weights, covariances)
    portfolio_risk = _portfolio_risk(weights, covariances)

    # Calculer assets_risk_contribution, la contribution de chaque actif au risque du portefeuille en utilisant la fonction _assets_risk_contribution_to_portfolio_risk(weights, covariances)
    assets_risk_contribution =_assets_risk_contribution_to_portfolio_risk(weights, covariances)
    # Calculer assets_risk_target (une matrice de dimension (nb_assets,1)), la contribution désiree de chaque actif au risque du portefeuille en multipliant portfolio_risk et asset_risk_budget
    assets_risk_target = np.asmatrix(np.multiply(portfolio_risk, assets_risk_budget))
    # Calculer l'erreur, nommee error) entre la contribution desiree (assets_risk_target) et la contribution realisee (assets_risk_contribution) de chaque actif
    error = sum(np.square(assets_risk_contribution - assets_risk_target.T))[0, 0]
    # La fonction _risk_budget_objective_error(weights, args) renvoie l'erreur calculee (nommee error)
    return error

# Creer la fonction _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights)
def _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights):
    # Creer constraints contenant 2 contraintes :
    # 1. Une egalite puisque la somme des poids est egale a 1
    # 2. Une inegalite puisque toute les positions sont longues, les poids sont donc superieur a 0
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                   {'type': 'ineq', 'fun': lambda x: x})

    # Creer optimize_result contenant les resultat de l'optimisation realisee avec la fonction minimize du package scipy.optimize
    # 1er argument : la fonction a optimiser
    # 2eme argument : les poids initiaux
    # 3eme argument : args contenant la matrice de covariance (covariances) et la contribution desiree de chaque actif au risque du portefeuille (assets_risk_budget)
    # 4eme argument : la methode d'optimisation
    # 5eme argument : les contraintes de constraints
    # 6eme argument : le parametre de tolerance pour l'arret de l'optimisation

    optimize_result = minimize(fun=_risk_budget_objective_error,
                               x0=initial_weights,
                               args=[covariances, assets_risk_budget],
                               method='SLSQP',
                               constraints=constraints,
                               tol=tolerance)

    # Recoperer dans weights les poids obtenus a la suite de l'optimisation (aller les chercher dans optimize_result)
    weights = optimize_result.x
    # La fonction _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights) renvoie les poids optimaux (nommee weights)
    return weights

# DEBUTER ICI VOTRE PROGRAMMATION
# Definir le 6eme parametre, nomme tolerance, le parametre de tolerance pour l'arret de l'optimisation est fixe a 1e-20
tolerance= 1e-20

# Definir le nombre d'actifs, nomme nb_assets, dans cet exemple ce nombre est egal a 4
nb_assets=4

# Definir la matrice de covariance, nommmee covariances, un array de dimension (nb_assets,nb_assets) d'apres l'enonce de l'Exercice 3
covariances = np.array([[0.01,0.02, 0.03, 0.04],[0.02,0.04, 0.06, 0.08],[0.03, 0.06, 0.09, 0.12],[0.04, 0.08, 0.12, 0.16]])


# Definir la contribution desiree de chaque actif au risque du portefeuille, nommee assets_risk_budget, un array de dimension (nb_assets,)
# Ici nous souhaitons que la contribution de chaque actif soit egal (la somme de ces contributions est egale a 1)
assets_risk_budget = np.full((4,1),[1 / nb_assets])



# Initialisation de la matrice des poids, nommee init_weights, un array de dimension (nb_assets,)
# Ici nous souhaitons debuter l'optimisation avec des poids equipondere
init_weights = np.full((4,1),[1 / nb_assets])

# Recuperer dans RiskParity_weights le resultat du processus d'optimisation sur les poids en appelant la fonction _get_risk_parity_weights(covariances, assets_risk_budget, init_weights)
RiskParity_weights = _get_risk_parity_weights(covariances, assets_risk_budget, init_weights)




"""""""""""""""
 Exercice 3
"""""""""""""""


warnings.filterwarnings('ignore')


"""""""""""""""""
question 1
"""""""""""""""""



#création de la dataframe
file=ExcelFile(r"Data4Exam_Ex3.xlsx")
df=file.parse('Data')
#regroupement par mois des données dans une dataframe grouped et calcul de la moyenne relative a chaque mois
date=df['TIME_PERIOD']
df['year_month'] = pd.to_datetime(df['TIME_PERIOD']).dt.to_period('M')
grouped = df.groupby('year_month').mean()
print('\n\n')



"""""""""""""""""
question 2
"""""""""""""""""
# yield curve à multiples dates

#choisir les annees 2004, 2007, 2008, 2009, 2012, 2015, et 2018
selections=[2004,2007,2008,2009,2012,2015,2018]

#création de variable pour trier les index
m=grouped.index.month
m.tolist()
y=grouped.index.year
y.tolist()

#tri des mois et années souhaitées, récupération des taux pour ces dates et création d'un graphique en fonction de la maturité
plt.figure(4)
for i in range(0,len(grouped)):
    for n in range(0,len(selections)):
        if m[i]==12 and y[i]==selections[n]:
            values=grouped.iloc[i, 0:len(grouped.columns)]
            values.tolist()
            headers=list(grouped.columns[0:len(grouped.columns)].values)
            plt.rcParams["figure.figsize"] = (30,12)
            plt.xlabel('maturities', fontsize=12)
            plt.ylabel('yield', fontsize=12)
            plt.plot(headers,values,label="yield curve "+str(m[i])+"-"+str(y[i]))
            plt.title('Yield Curve for several dates', fontsize=16)
            plt.legend(loc="top right",bbox_to_anchor=(1.1, 1.05), fontsize=12)
plt.show()

print("réponse Ex3 question 2:\n\
      Une politique d'augmentation des taux spots n'a été observable qu'entre 2004 et 2007 \
avec des accroissements plus prononcés à de courtes maturités. La crise financière de l'année 2008 \
se caractérise par une rechute des taux à courtes maturités, créant une courbe des taux presque similaire à 2004. \
En 2009, les maturités de court terme sont à nouveau revues à la baisse de manière significative avec des différences allant jusqu'à \
plus de 150 points de base. En 2012, les taux relatifs aux longues maturités ont été réajustés à la baisse à près de 200 points de base \
sans doute dans l'objectif d'aplanir à nouveau la courbe des taux, rapprochant ainsi les valeurs des taux des maturités de moyen et long terme. \
Puis une baisse plus homogène des taux toute maturité confondue a pris place entre 2015 et 2020, \
faisant progressivement chuter la courbe des taux dans le négatif \n\
    L’effet significatif de la politique monétaire européenne sur la courbe des taux a contribué \
à améliorer les conditions de financement et à assouplir l’orientation de la politique monétaire. \
En même temps, le fort impact de cette politique rend plus difficile la lecture de l’information \
que la courbe des taux peut intégrer en ce qui concerne les perspectives pour l’économie. Un exemple important \
et actuellement pertinent est la question de savoir dans quelle mesure une courbe des taux qui s’aplanit signale un \
affaiblissement des perspectives économiques.L'intérêt doit ainsi être centré sur la corrélation négative entre la pente \
de la courbe des taux et la probabilité de récessions futures, ainsi que sur les mécanismes économiques possibles sous-tendant une \
telle configuration.",'\n\n\n')

"""""""""""""""""
question 3
"""""""""""""""""           
#echantillons
echantillon1=grouped['2004-09':'2008-08']
echantillon2=grouped['2008-09':'2019-12']


#necessary functions
normalize=lambda x: (x-x.mean())/x.std()
fractions=lambda x: x/x.sum() 


#nombre de composantes principales echantillon 1
pca1=PCA().fit(echantillon1.apply(normalize))
pca_components1=pca1.fit_transform(echantillon1)
pca1.explained_variance_ratio_

#nombre de composantes principales echantillon 2
pca2=PCA().fit(echantillon2.apply(normalize))
pca_components2=pca2.fit_transform(echantillon2)
pca2.explained_variance_ratio_



"""""""""""""""""
question 4
"""""""""""""""""

#calcul des vecteurs propres de l'échantillon 1
cov=np.cov(echantillon1.T)
eigen_values, eigen_vectors = np.linalg.eig(cov)
#représentation des trois premiers vecteurs propres
plt.figure(5)
plt.plot(echantillon1.columns, eigen_vectors.T[0], label="PCA_1")
plt.plot(echantillon1.columns, eigen_vectors.T[1],label="PCA_2")
plt.plot(echantillon1.columns, eigen_vectors.T[2],label="PCA_3")
plt.legend(loc='best', fontsize=12)
plt.xlabel('maturities', fontsize=12)
plt.ylabel('eigenvectors', fontsize=12)
plt.title('PCA vectors sample 1', fontsize=16)
plt.rcParams["figure.figsize"] = (30,12)
plt.show()


#calcul des vecteurs propres de l'échantillon 2
cov=np.cov(echantillon2.T)
eigen_values, eigen_vectors = np.linalg.eig(cov)
#représentation des trois premiers vecteurs propres
plt.figure(6)
plt.plot(echantillon2.columns, eigen_vectors.T[0], label="PCA_1")
plt.plot(echantillon2.columns, eigen_vectors.T[1],label="PCA_2")
plt.plot(echantillon2.columns, eigen_vectors.T[2],label="PCA_3")
plt.legend(loc='best', fontsize=12)
plt.xlabel('maturities', fontsize=12)
plt.ylabel('eigenvectors', fontsize=12)
plt.title('PCA vectors sample 2', fontsize=16)
plt.rcParams["figure.figsize"] = (30,12)
plt.show()

print("réponse Ex3 question 4:\n\
      L'analyse en composante principale est une métode qui permet d'identifier \
les logiques dominantes relatives aux mouvements simultanés des points sur une \
courbe des taux. Les 3 premières composantes sont simplement les 3 premières dimensions de l'analyse \
en composantes principales qui capturent la plupart de la variance des données. \
La plus grande variance provient d’un décalage parallèle dans la courbe, \
la seconde d’une inclinaison de la courbe, et la troisième d’une flexion de la courbe. \
Il n’est pas nécessaire qu’il en soit ainsi, c’est seulement la dynamique du \
marché qui peut être identifiée avec les composantes principales.\
Seuls les trois premiers vecteurs propres sont représentés dans le but de \
ne garder que les composantes principales représentant les variables les plus \
corrélées. \n\
Dans notre cas, il est possible d'observer que pour nos deux échantillons:\n\
-le premier vecteur propre a tous ses composents négatifs \n\
-le second identifie que les taux entre 0 et 10 ans bouge dans une direction négative \
et le reste dans une direction positive. Il s'agit d'une torsion \n\
-le troisième surligne que les taux à faibles et fortes maturités bouger dans une direction négative\
Les maturités moyennes,elles, bougent dans une direction positive. Il s'agit d'une courbure \n\
Ainsi, la représentation des vecteurs propres permet de suivre \
la manière dont les taux à différentes maturités bougent ensemble sur la courbe des taux,\
ces mouvements pouvant être identiques ou contraires en fonction des périodes."
, '\n\n\n')



"""""""""""""""""
question 5
"""""""""""""""""   

#ajout de la yield curve de Mars 2020 dans le graphique contennant toutes les yield curves
yield_2020=plt.figure(7)
for i in range(0,len(grouped)):
    if m[i]==3 and y[i]==2020:
        values_mars2020=grouped.iloc[i, 0:len(grouped.columns)]
        values_mars2020.tolist()
        headers=list(grouped.columns[0:len(grouped.columns)].values)
        plt.plot(headers,values_mars2020,label="yield curve "+str(m[i])+"-"+str(y[i]))
        plt.legend(loc="top right",bbox_to_anchor=(1.1, 1.0), fontsize=12)
plt.show()

print('\n',"réponse Ex3 question 5:\n\
      Le COVID-19 a provoqué une diminution de la courbe des taux à moyen et long terme, \
comme on peut l’observer en mars 2020. Cette diminution des taux a pour but de parer\
à la déflation et au manque de liquidité liés au ralentissement des économies européennes. \
Faire face au virus nécessite d’aider les pays à emprunter à des échéances très longues et à des taux d’intérêts \
bas tout en partageant la partie asymétrique des coûts. Le taux actuellement en vigueur pour les \
emprunts d'Etat à dix ans est ainsi à son niveau le plus bas observé depuis octobre 2019. \
Si emprunter n’a jamais couté aussi peu pour les états, c’est qu’un besoin monétaire urgent \
s’est fait ressentir. L’augmentation des taux d’emprunts pour certains états membres est ainsi justifiée, \
car leur équilibre financier en dépend.",' \n\n\n')

  