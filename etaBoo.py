# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:38:18 2020

@author: jonathan beaulieu-emond
"""
import numpy as np
from math import pi
from numba import jit,prange,njit
from math import pi
import matplotlib.pyplot as plt

#Variable globale
tj,Vobs,w=np.loadtxt('etaBoo.txt',unpack=True)
#tj,Vobs,w=np.loadtxt('rhoCorB.txt',unpack=True)
sigma=1/w

@njit
def frontiere(u,L,N):
    """
    

    Parameters
    ----------
    u : array containing the different parameters of the population.
    L : Condition frontieres, impair->lower limit,pair->upper limit.
    N :Nombre de paramètres pour chaque individu.

    Returns
    -------
    u : Return the original array with the parameters now bounded by L.

    """
    #redistribuer de manière proportionelle à sa sortie/aléatoire? # ne marche pas pour de petites distribution...
    for i in range(0,N) :
        u[i,:]=np.where(u[i,:]<L[2*i],np.random.uniform(L[2*i],L[2*i+1],len(u[i,:])),u[i,:])
        u[i,:]=np.where(u[i,:]>L[2*i+1],np.random.uniform(L[2*i],L[2*i+1],len(u[i,:])),u[i,:])
    return u
        
    
        
def diffrac2d(u) :
    x,y=u
    return (np.sin(x)/x)**2*(np.sin(y)/y)**2

    
        
def diffrac5d(u) :
    x,y,z,a,b=u
    return (np.abs((np.sin(x+4)/(x+4))**2*(np.sin(y-3)/(y-3))**2*(np.sin(z)/(z))**2*(np.sin(a)/(a))**2*(np.sin(b)/(b))**2))
        
     
@jit
def bissec(f,x1,x2,epsilon,param) :
    """
    

    Parameters
    ----------
    f :function to find roots.
    x1 :lower bound.
    x2 : upper bound.
    epsilon : preision of the results.
    param : tuple of the parameters required for the function.The function must accept a tuple as second argument
    Returns
    -------
    root : return, if possible, a root of the function found between x1 and x2.

    """
   
    itermax=400
    delta=x2-x1
    k=0
    while (delta>epsilon) and k<=itermax :
        xm=0.5*(x1+x2)
        fm=f(xm,param)
        f2=f(x2,param)
        if fm*f2>0 :
            x2=xm
        else :
            x1=xm
        delta=x2-x1
        k+=1
    return xm
        
        
@njit
def anomalie_excentrique(E,param) :
    (ecc,P,tau,tj)=param
    return E-ecc*np.sin(E)-2*pi/P*(tj-tau)
             


@njit
def vkepler(u) :
    p,tau,w,e,k,V0=u
    #Paramètres de la bissection
    eps=1e-6
    chi2=0
    
   
    for i in range(0,len(tj)) :
        E=bissec(anomalie_excentrique,-300,300,eps,(e,p,tau,tj[i]))
        v=2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
       
        V=V0+k*(np.cos(w+v))+k*e*np.cos(w)
        chi2+=1/sigma[i]**2*(Vobs[i]-V)**2
    chi2/=(len(tj)-6)
   
    return np.true_divide(1,chi2)

       
@njit    
def roulette(rang,N) :
    """
    Algorithme de la roulette pour la pression sélective de l'algorithme génétique'

    Parameters
    ----------
    rang : classement de chaque individu de la population.
    N : Nombre de paramètres.

    Returns
    -------
    k : index de l'individu sélectionné pour la reproduction.

    """
   
    r=N*(N+1)/2*np.random.uniform(0,1)
    scumul=0
    for  k in range(0,N) :
        scumul+=(1+rang[k])
        if scumul>=r :   return k
          


@jit
def Tinder(u,rang,taille,pc,N,pm,ecart,sigma,multiplicatif) :
    """
    

    Parameters
    ----------
    u : Tableau des individus.
    rang : Classements des individus selon la fonctio mérite.
    taille :Nombre d'individu.
    pc : Probabilité de croisement.
    N : Nombre de paramètres.
    pm : probabilité de mutation.
    ecart : range de mutation défini par la taille du domaine exploré de chaque paramètre .

    Returns
    -------
    kids1 : premiere moitié de la nouvelle génération d'individu.
    kids2 : seconde moitié de la nouvelle génération d'individu.

    """
    #initialisation des tableaux et variables
    parent=np.int(taille/2)
    i1,i2=np.zeros(parent,dtype=np.int32),np.zeros(parent,dtype=np.int32) # tableau des indices des papas et mamans
    for k in range(0,parent) :
            
        i1[k]=roulette(rang,taille)
        i2[k]=i1[k]
        while(i2[k]==i1[k]) :
            i2[k]=roulette(rang,taille) #s'assure que les parents sont différent
        
       
    #--------------------------------------------------
    #Cue Careless Whispers (faut vrm que j'explique...?)
    
    lucky_dudes=np.copy(u[:,i1]) # les papas (aka leur valeur)
    lucky_ladies=np.copy(u[:,i2]) # les mamans
    
    r1=np.random.uniform(size=(N,parent)) # ratio de genes de maman vs papa
    r2=np.random.uniform(size=(N,parent))
    
    
    cock_block=np.random.uniform(size=parent) #probilité d'avoir un enfant ou non
    kids1=lucky_ladies*r1+(1-r1)*lucky_dudes #1er enfant
    kids1=np.where(cock_block<pc,kids1,u[:,i1])
    
    #cock_block=np.random.uniform(size=parent) #probilité d'avoir un enfant ou non
    kids2=lucky_ladies*(1-r2)+r2*lucky_dudes # 2e enfant
    kids2=np.where(cock_block<pc,kids2,u[:,i2]) 
   
    
    #---------------------------------------------
    #Mutation des gènes (On échappe les kids oops)
    r1=np.random.uniform(size=(N,parent))# pour la mutation
    r2=np.random.uniform(size=(N,parent))# pour la mutation
    #sigma=1000#????
   
    #kids1[:,i]+=np.where(np.logical_and(r1[i]<pm,cock_block[i]<pc),np.random.normal(0,ecart/sigma,size=((N))),0)
    #kids2[:,i]+=np.where(np.logical_and(r2[i]<pm,cock_block[i]<pc),np.random.normal(0,ecart/sigma,size=((N))),0)
    cock_block=np.array([cock_block,]*N)
    kids1+=np.where(np.logical_and(r1<pm,cock_block<pc),multiplicatif*np.random.normal(0,ecart/sigma,size=((N,parent))),0)
    kids2+=np.where(np.logical_and(r2<pm,cock_block<pc),multiplicatif*np.random.normal(0,ecart/sigma,size=((N,parent))),0)
    kids=np.concatenate((kids1,kids2),axis=1)
        #ajouter une variable r pour chaque parametre#!!!!!
    return kids

@njit
def elitisme(u,chi2,rang,bestx,bestf,taille,N) :
    
    #Fonction sauvegardant la meilleure solution de chaque génération
    u[:,rang[0]]=bestx
    chi2[rang[0]]=bestf
    rang=np.argsort(chi2)
    
    bestx=np.copy(u[:,rang[taille-1]])
    bestf=chi2[rang[taille-1]]
   
    """
    popt=np.where(chi2==np.min(chi2))[0]
    if len(popt)>1 :
        popt=popt[0]
    u[:,popt]=bestx
    chi2[popt]=bestf
    popt=np.where(chi2==np.max(chi2))[0]
    if len(popt)>1 :
        popt=popt[0]
    bestx=u[:,popt]
    bestf=chi2[popt]
    """
    return bestx,bestf,u,rang
@njit
def adaptatif(pm,delta,sigma,compteur,multiple) :
    #Version 0.2 de la mutation adaptative, à améliorer
    if delta<0.009 :
        compteur+=1
    if compteur>30 :
        pm+=0.1
        multiple=3
        compteur=0
    else :
        multiple=1
   # if delta>10 and pm>0.01:
    #    pm=-0.005
    """
    if pm>10 :
        pm=0.01
        if sigma-0.01>0.01 :
            sigma-=0.01
    """
    """
    if pm>1 :
        sigma-=0.1
    if pm<0.1 :
        sigma=10
    """
    return pm,sigma,compteur,multiple
@jit 
def algorithme_genetique(f,N,M,L,param) :
    """
  

    Parameters
    ----------
    f : function that you want to optimize.Must return a value to evaluate the fit and take an array of parameter as argument.
    N : Number of variables to optimize.
    M : +1 to maximize or -1 to minimize.
    L : limits of parameter search for each variable as an array [xmin,xmax,ymin,ymax,....]
    param : parameters (pm,pc,taille,itermax)
    Returns
    -------
    Return the fitted value for each parameter in the order the function f takes them.

    """
    #-----------------------------
    #Paramètre du modèle
    pm,pc,taille,iterMax=param
    compteur=0
    L=np.array(L)
    ecart=np.array(np.abs(L[::2]-L[1::2]))
    ecart=np.array([ecart,]*int(taille/2)).transpose()
    
    #------------------------------
    #Initialisation des tableaux
    u=np.zeros((N,taille))
   
    j=0
    
    for i in range(0,N) :
        u[i,:]=np.random.uniform(L[j],L[j+1],taille)
        j+=2
        
    futur_u=np.copy(u)
    chi2=np.zeros(taille,dtype=np.float64)
    rang=np.zeros(taille,dtype=np.int64)
    bestf=0
    bestx=u[:,1]
    bestbestf=0
    multiple=1
    #----------------------------
    #Boucle sur les generations
    delta=1
    time=0
    sigma=20
    iter=0
    while iter<iterMax : 
         iter+=1
         
        # sigma=iter*10 #????
         # Vérification des frontières
         u=np.copy(futur_u)
         u=frontiere(u,L,N)
         #-----------------------------
         #Évaluation de notre population
         for i in range(0,taille) :
             chi2[i]=f(u[:,i])
            
         rang=np.argsort(chi2)
        
         #-----------------------------
         #Élitisme
         bestx,bestf,u,rang=elitisme(u,chi2,rang,bestx,bestf,taille,N)
         
         #évaluation de l'erreur 
         delta=chi2[rang[taille-1]]-chi2[rang[np.int(taille/2)]]
         #delta=np.abs(np.sum((u[:,rang[taille-1]]-u[:,rang[int(taille/2)]])/u[:,rang[taille-1]]))
         print('generation=',(iter),'chi2=',chi2[rang[taille-1]],np.mean(chi2),u[:,rang[taille-1]],'mutation=',(pm),'delta=',(delta))
             
         #-----------------------------
         #-----------------------------
         #Adaptation de la mutation
         
         pm,sigma,compteur,multiple=adaptatif(pm,delta,sigma,compteur,multiple)
         #pm=(np.sin(iter/10))**2
         """
         if iter%100==0 :
             pm=0.9
             j=0
             for i in range(0,N) : # the PURGE
                 u[i,0:int(1/10*taille)]=np.random.uniform(L[j],L[j+1],int(1/10*taille))
                 j+=2
             for i in range(0,taille) :
                 chi2[i]=f(u[:,i])
            
             rang=np.argsort(chi2)
         """
         """
         if delta/chi2[rang[taille-1]]>0.9 :
              sigma-=0.1
         else :
             sigma=10
         """
         """
         if  bestf==bestbestf :
    
             if sigma>3 : sigma/=1.25
         else :
             bestbestf=bestf
             sigma=10
         """
         #-----------------------------
         #Reproduction
         kids=Tinder(u,rang,taille,pc,N,pm,ecart,sigma,multiple)
         
         futur_u[:,:]=np.copy(kids)
         
         #-----------------------------
         
         
    return [u[:,rang[taille-1]]],chi2[rang[taille-1]]


#%%
L=[-3*pi,3*pi,-3*pi,3*pi,-3*pi,3*pi,-3*pi,3*pi,-3*pi,3*pi]
ubest=algorithme_genetique(diffrac5d,5,+1,L,(0.1,0.8,10,600))
print(ubest)
#%%
t0=np.min(tj)
Kmax=np.max(Vobs)-np.min(Vobs)
L=(200,800,t0,t0+800.,0.,2*pi,0.,1.,0.,Kmax,np.min(Vobs),np.max(Vobs))
#L=(400.,600.,t0,t0+800.,1.,3,0.,1,0.,Kmax,np.min(Vobs),np.max(Vobs))
#param=(0.1,0.8,10,100)
chiff=[]
#for l1,l2 in zip([0,10,20,30,40,50,60,70,80],[10,20,30,40,50,60,70,80,90]) :
    #L=(l1,l2,t0,t0+l2,0.,2*pi,0.,1.,0.,Kmax,np.min(Vobs),np.max(Vobs))
ubest,chi2=algorithme_genetique(vkepler,6,+1,L,(0.1,0.8,20,6000))
#    chiff.append(chi2)

#u=[494.20,14299,5.7397,0.2626,8.3836,1.0026] chi2=0.212961
fig, axs = plt.subplots(2)
axs[1].set_xlabel('J.D')
axs[0].set_ylabel(r'V[km $s^{-1}$]')
axs[1].set_ylabel(r'V[km $s^{-1}$]')
def graph(uu,lable,fig,axs) :
    p,tau,w,e,k,V0=uu
    #Paramètres de la bissection
    eps=1e-8
   
    
    tjj=np.arange(np.min(tj),tj[33])
    V=np.zeros(len(tjj))
    for i in range(0,len(tjj)) :
        E=bissec(anomalie_excentrique,-300,300,eps,(e,p,tau,tjj[i]))
        v=2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
        V[i]=V0+k*(np.cos(w+v))+e*np.cos(w)
   
    axs[0].plot(tjj,V,label=lable)
    axs[0].errorbar(tj[0:34],Vobs[0:34],yerr=sigma[0:34],fmt='.',label='Observé')
    axs[0].legend()
    
    tjj=np.arange(tj[38],np.max(tj))
    V=np.zeros(len(tjj))
    for i in range(0,len(tjj)) :
        E=bissec(anomalie_excentrique,-300,300,eps,(e,p,tau,tjj[i]))
        v=2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
        V[i]=V0+k*(np.cos(w+v))+e*np.cos(w)
    axs[1].plot(tjj,V,label=lable)
    axs[1].errorbar(tj[38:len(tj)],Vobs[38:len(tj)],yerr=sigma[38::],fmt='.',label='Observé')
    axs[1].legend()
   
  
graph(ubest[0],'solution',fig,axs)

graph([494.20,14299,5.7397,0.2626,8.3836,1.0026],'prof',fig,axs)

"""
p,tau,w,e,k,V0=ubest
tjj=np.arange(np.min(tj),np.max(tj))
V=np.zeros(len(tjj))
for i in range(0,len(tjj)) :
        E=bissec(anomalie_excentrique,-300,300,1e-7,(e,p,tau,tjj[i]))
        v=2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
        V[i]=V0+k*(np.cos(w+v))+e*np.cos(w)
plt.plot(tjj,V,)
plt.plot(tj,Vobs,'.')
"""
#%%
L=[-3*pi,3*pi,-3*pi,3*pi]
ubest=algorithme_genetique(diffrac2d,2,+1,L,(0.1,0.8,10,100))
print(ubest)

