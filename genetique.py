# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:38:18 2020

#@author: jonathan beaulieu-emond
"""
import numpy as np
from math import pi
from numba import jit,prange,njit
from math import pi
import matplotlib.pyplot as plt
import random
#Variable globale
tj,Vobs,w=np.loadtxt('etaBoo.txt',unpack=True)
#tj,Vobs,w=np.loadtxt('rhoCorB.txt',unpack=True)
sigma=1/w



#@njit
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
    for i in range(0,N) :
        u[i,:]=np.where(u[i,:]<L[2*i],L[2*i],u[i,:])
        u[i,:]=np.where(u[i,:]>L[2*i+1],L[2*i+1],u[i,:])
       
    return u
        
    
        
def diffrac2d(u) :
    x,y=u
    return (np.sin(x)/x)**2*(np.sin(y)/y)**2

    
        
def diffrac5d(u) :
    x,y,z,a,b=u
    return (np.abs((np.sin(x+4)/(x+4))**2*(np.sin(y-3)/(y-3))**2*(np.sin(z)/(z))**2*(np.sin(a)/(a))**2*(np.sin(b)/(b))**2))
        
     
#@jit
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
   
    itermax=100
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
        
        
#@njit
def anomalie_excentrique(E,param) :
    (ecc,P,tau,tj)=param
    return E-ecc*np.sin(E)-2*pi/P*(tj-tau)
             


#@njit
def vkepler(u) :
    p,tau,w,e,k,V0=u
    #Paramètres de la bissection
    eps=1e-5
    chi2=0
    
   
    for i in range(0,len(tj)) :
        E=bissec(anomalie_excentrique,-2000,2000,eps,(e,p,tau,tj[i]))
        v=2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
       
        V=V0+k*(np.cos(w+v))+k*e*np.cos(w)
        chi2+=1/sigma[i]**2*(Vobs[i]-V)**2
    chi2/=(len(tj)-6)
   
    return np.true_divide(1,chi2)

       
#@njit    
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
        scumul+=1+rang[k]
        if scumul>=r :
            return k
###@jit

#@jit
def Tinder(u,rang,taille,pc,N,pm,ecart,sigma,LLL) :
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
    
    lucky_dudes=u[:,i1] # les papas (aka leur valeur)
    lucky_ladies=u[:,i2] # les mamans
    
    kids1=np.zeros((N,parent))
    kids2=np.zeros((N,parent))
    LL=100000000 
    precision=8
    
    for j in range(0,parent) :
       
        cock_block=np.random.uniform() #probilité d'avoir un enfant ou non
        if cock_block<pc :
            papa=''
            maman=''
           
            for i in range(0,N) :
                if lucky_dudes[i,j]*LL<10**(precision-1) :
                    papa+='0'*(precision-1)+'1'
                   
                else :
                    papa+=str((lucky_dudes[i,j]*LL))[0:precision]
                  
                if lucky_ladies[i,j]*LL<10**(precision-1) :
                    maman+='0'*(precision-1)+'1'
                   
                else :
                    maman+=str((lucky_ladies[i,j]*LL))[0:precision]
                   
                
            
            r=random.randint(0,N*precision)
            kids11=str(papa)[0:r]+str(maman)[r::]
            kids22=str(maman)[0:r]+str(papa)[r::]
            if np.random.uniform()<pm :
                r=random.randint(0,N*precision)
                kids11=kids11[0:r]+str(random.randint(0,9))+kids11[r+1::]
                r=random.randint(0,N*precision)
                kids22=kids22[0:r]+str(random.randint(0,9))+kids22[r+1::]
                
                
           
            else :
                   kids1[i,j]=int(kids1[i,j])
                   kids2[i,j]=int(kids2[i,j])
                   
                
            for i in range(0,N) :
                print(precision-LLL[i])
                kids1[i,j]=int(kids11[i*precision:(i+1)*precision])/10**(precision-LLL[i])
                kids2[i,j]=int(kids22[i*precision:(i+1)*precision])/10**(precision-LLL[i])
                
          
        else :
           r=np.random.uniform()
           papa=''
           maman=''
           Lpapa=[]
           Lmaman=[]
           for i in range(0,N) :
                if lucky_dudes[i,j]*LL<10**(precision-1) :
                    papa+='0'*precision
                    Lpapa.append(1)
                else :
                    papa+=str((lucky_dudes[i,j]*LL))[0:precision]
                    Lpapa.append(len(str((int(lucky_dudes[i,j]*LL)))[precision::]))
                if lucky_ladies[i,j]*LL<10**(precision-1) :
                    maman+='0'*precision
                    Lmaman.append(1)
                else :
                    maman+=str((lucky_ladies[i,j]*LL))[0:precision]
                    Lmaman.append(len(str((int(lucky_ladies[i,j]*LL)))[precision::]))
           kids11=papa
           kids22=maman
           if r<pm :
               r=random.randint(0,len(papa))
               kids11=str(kids11)[0:r]+str(random.randint(0,9))+str(kids11)[r+1::]
               r=np.random.randint(0,len(maman))
               kids22=str(kids22)[0:r]+str(random.randint(0,9))+str(kids22)[r+1::]
           for i in range(0,N) :
                kids1[i,j]=int(kids11[i*precision:(i+1)*precision])/LL*10**np.int(np.mean([Lpapa[i],Lmaman[i]])-1)
                kids2[i,j]=int(kids22[i*precision:(i+1)*precision])/LL*10**np.int(np.mean([Lpapa[i],Lmaman[i]])-1)
           
  
    return kids1,kids2

#@jit
def elitisme(u,chi2,rang,bestx,bestf,taille,N) :
    
    #Fonction sauvegardant la meilleure solution de chaque génération
    u[:,rang[0]]=bestx
    chi2[rang[0]]=bestf
    rang=np.argsort(chi2)
    
    bestx=np.copy(u[:,rang[taille-1]])
    bestf=np.copy(chi2[rang[taille-1]])
   
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
#@njit
def adaptatif(pm,delta,eps_min,eps_max) :
    #Version 0.2 de la mutation adaptative, à améliorer
    if delta<1 :
        pm+=0.005
    if delta>10 and pm>0.01:
        pm=-0.005
    return pm
#@jit 
def algorithme_genetique(f,N,M,L,param,adaptation,LL) :
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
    eps_min=0.1
    eps_max=0.4
    epsilon=2.3
    L=np.array(L)
    ecart=np.abs(L[::2]-L[1::2])
    
    #------------------------------
    #Initialisation des tableaux
    u=np.zeros((N,taille),dtype=np.int32)
   
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
    chimean=[]
    #----------------------------
    #Boucle sur les generations
    delta=1
    time=0
    sigma=10
    iter=0
    while iter<iterMax : 
         iter+=1
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
         #-----------------------------
         #Adaptation de la mutation
         if adaptation :
             pm=adaptatif(pm,delta,eps_min,eps_max)
         """
         if  bestf==bestbestf :
    
             if sigma>3 : sigma/=1.25
         else :
             bestbestf=bestf
             sigma=10
         """
         #-----------------------------
         #évaluation de l'erreur 
         delta=np.abs(np.sum((u[:,rang[taille-1]]-u[:,rang[int(taille/2)]])/u[:,rang[taille-1]]))
         print('generation=',(iter),'chi2=',chi2[rang[taille-1]],u[:,rang[taille-1]],'mutation=',(pm),'delta=',(delta))
         #print(u[:,rang[taille-1]])
         #-----------------------------
         #Reproduction
         kids1,kids2=Tinder(u,rang,taille,pc,N,pm,ecart,sigma,LL)
         
         futur_u[:,0:int(taille/2)]=np.copy(kids1[:,:])
         futur_u[:,int(taille/2)::]=np.copy(kids2[:,:])
         #chimean.append(1-chi2[rang[taille-1]])
         #-----------------------------
         
         
    return u[:,rang[taille-1]]


#%%
L=[-3*pi,3*pi,-3*pi,3*pi,-3*pi,3*pi,-3*pi,3*pi,-3*pi,3*pi]
ubest=algorithme_genetique(diffrac5d,5,+1,L,(0.05,0.8,10,5),False)
print(ubest)
#%%
t0=np.min(tj)
Kmax=np.max(Vobs)-np.min(Vobs)
L=(200.,800.,t0,t0+800.,0.,2*pi,0.,0.999999,0.,Kmax,np.min(Vobs),np.max(Vobs))
LL=(2,4,1,0,1,1)
param=(0.1,0.8,10,100)
ubest=algorithme_genetique(vkepler,6,+1,L,(1,0.8,10,6000),False,LL)

#u=[494.20,14299,5.7397,0.2626,8.3836,1.0026] chi2=0.212961
def graph(uu,lable) :
    p,tau,w,e,k,V0=uu
    #Paramètres de la bissection
    eps=1e-8
   
    
    tjj=np.arange(np.min(tj),tj[33])
    V=np.zeros(len(tjj))
    for i in range(0,len(tjj)) :
        E=bissec(anomalie_excentrique,-300,300,eps,(e,p,tau,tjj[i]))
        v=2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
        V[i]=V0+k*(np.cos(w+v))+e*np.cos(w)
   
    plt.plot(tjj,V,label=lable)
    plt.plot(tj[0:34],Vobs[0:34],'.')
    plt.legend()
    plt.show()
    tjj=np.arange(tj[38],np.max(tj))
    V=np.zeros(len(tjj))
    for i in range(0,len(tjj)) :
        E=bissec(anomalie_excentrique,-300,300,eps,(e,p,tau,tjj[i]))
        v=2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
        V[i]=V0+k*(np.cos(w+v))+e*np.cos(w)
    plt.plot(tjj,V,label=lable)
    plt.plot(tj[38:len(tj)],Vobs[38:len(tj)],'.')
    plt.legend()
    plt.show()
  
graph(ubest,'best')

graph([494.20,14299,5.7397,0.2626,8.3836,1.0026],'prof')
#%%
L=[-3*pi,3*pi,-3*pi,3*pi]
for pm,adapt in zip([0.1,0.25,0.5,0.1,0.25,0.5],[True,True,True,False,False,False]) :
    ubest,umean=algorithme_genetique(diffrac2d,2,+1,L,(pm,0.8,10,100),adapt)
    plt.plot(np.arange(0,100),umean,label='pm='+str(pm)+',adapt='+str(adapt))
print(ubest)
plt.legend()
plt.xlabel('generation')
plt.ylabel(r'$\epsilon$')
plt.semilogy()
plt.show()
