#funcion que calcula la distancia entre un cuerpo y todos los demas
import numpy as np
import numba as nm
from numba import jit

@jit(nopython=True)
def distancia(x,y,i,j):
    distancia=np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
    return distancia


#función que calcula la fuerza en x
@jit(nopython=True)
def fuerza_x(x,y,M,f_x):
    for i in range(0,len(x)):
        suma=0.0
        for j in range(0,len(x)):
            if i!=j:
                suma=suma-M[j]*(x[i]-x[j])/distancia(x,y,i,j)**3
        f_x[i]=suma
    return f_x
    
#función que calcula la fuerza en y
@jit(nopython=True)
def fuerza_y(x,y,M,f_y):
    for i in range(0,len(x)):
        suma=0.0
        for j in range(0,len(x)):
            if i!=j:
                suma=suma-M[j]*(y[i]-y[j])/distancia(x,y,i,j)**3
        f_y[i]=suma
    return f_y

#funciones que calculan las posiciones nuevas en x e y
@jit(nopython=True)
def posnueva_x(x,vx,f_x,d,i,c,h):
    if d<550e9/c:
        return x[i]+h*vx[i]+0.5*h**2*f_x[i]
    if d>550e9/c:
        hr=8*h
        return x[i]+hr*vx[i]+0.5*hr**2*f_x[i]

@jit(nopython=True)
def posnueva_y(y,vy,f_y,d,i,c,h):
    if d<550e9/c:
        return y[i]+h*vy[i]+0.5*h**2*f_y[i]
    if d>550e9/c:
        hr=8*h
        return y[i]+hr*vy[i]+0.5*hr**2*f_y[i]

#funciones que calculas las nuevas w
@jit(nopython=True)    
def w_nueva_x(vx,f_x,d,i,c,h):
    if d<550e9/c:
        return vx[i]+0.5*h*f_x[i]
    if d>550e9/c:
        hr=8*h
        return vx[i]+0.5*hr*f_x[i]

@jit(nopython=True)
def w_nueva_y(vy,f_y,d,i,c,h):
    if d<550e9/c:
        return vy[i]+0.5*h*f_y[i]
    if d>550e9/c:
        hr=8*h
        return vy[i]+0.5*hr*f_y[i]

#funciones que claculan las nuevas velocidades
@jit(nopython=True)
def vel_nueva_x(w_x,f_x,d,i,c,h):
    if d<550e9/c:
        return w_x[i]+0.5*h*f_x[i]
    if d>550e9/c:
        hr=8*h
        return w_x[i]+0.5*hr*f_x[i]
@jit(nopython=True)
def vel_nueva_y(w_y,f_y,d,i,c,h):
    if d<550e9/c:
        return w_y[i]+0.5*h*f_y[i]
    if d>550e9/c:
        hr=8*h
        return w_y[i]+0.5*hr*f_y[i]
    
@jit(nopython=True)
def actualizar_estado(x,y,vx,vy,w_x,w_y,f_x,f_y,M,c,h):
    for k in range(0,len(x)):
         d=distancia(x,y,k,0)
         w_x[k]=w_nueva_x(vx,f_x,d,k,c,h)
         w_y[k]=w_nueva_y(vy,f_y,d,k,c,h)
         x[k]=posnueva_x(x,vx,f_x,d,k,c,h)
         y[k]=posnueva_y(y,vy,f_y,d,k,c,h)
         f_x=fuerza_x(x,y,M,f_x)
         f_y=fuerza_y(x,y,M,f_y)
         vx[k]=vel_nueva_x(w_x,f_x,d,k,c,h)
         vy[k]=vel_nueva_y(w_y,f_y,d,k,c,h)
        
    

#funciones que calculan la energía cinética y potencial
@jit(nopython=True)
def Ec(vx,vy,M,T):
    for i in range(0,len(vx)):
        T[i]=0.5*M[i]*(vx[i]**2+vy[i]**2)
    return T

@jit(nopython=True)
def Ep(x,y,M,V):
    for i in range(0,len(x)):
        suma=0.0
        for j in range(0,len(x)):
            if i!=j:
                suma=suma-M[i]*M[j]/distancia(x,y,i,j)

        V[i]=suma
    return V

#funcion que calcula el momento angular
@jit(nopython=True)
def momento_angular(x,y,vx,vy,M,L):
    for i in range(len(x)):
        L[i]= distancia(x,y,i,0)*M[i]*np.sqrt(vx[i]**2+vy[i]**2) 
    return L

@jit(nopython=True)
def Energía_total(E):
    for i in range(0,len(E)):
        ET=np.sum(E)
    return ET

#funcion que calcula el momento angular total
@jit(nopython=True)
def momento_angular_total(L):
    for i in range(len(L)):
        LT=np.sum(L)
    return LT




