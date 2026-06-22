import numpy as np
import numba as nm
from numba import jit
from funciones import distancia
from funciones import fuerza_x
from funciones import fuerza_y

m=1.989e30
c=1.49e11
M=np.array([1.0,0.33e24/m], dtype=np.float64) #masas del sol y de mercurio
x=np.array([0.0,57.9e9/c],dtype=np.float64)#posiciones en el eje x
y=np.array([0.0,0.0],dtype=np.float64)#posiciones en el eje y
f_x=np.zeros(len(x))
f_y=np.zeros(len(x))

x=np.array([0.0,57.9e9/c,108.2e9/c,149.6e9/c,228e9/c,778.5e9/c,1432e9/c,2867e9/c,4515e9/c],dtype=np.float64)#posiciones en el eje x
x=np.array([0.0,69.8e9/c,108.9e9/c,152.1e9/c,249e9/c,816.4e9/c,1506.5e9/c,3001.4e9/c,4558.9e9/c],dtype=np.float64)#posiciones en el eje x, Afelio

#f_x=fuerza_x(x,y,M,np.zeros(len(x)))
#f_y=fuerza_y(x,y,M,np.zeros(len(x)))
#print(f_x,f_y)
#importante: como agregar elemnetos a una matriz:
#R=np.array([[[1,2,3,4],[5,7,8,2]],[[16,2,3,4],[5,77,8,2]]])
#print(R)
#P=np.insert(R,len(R),[0,0,0,0],axis=0)
#print(P)
z=np.empty([0,len(x)])
a=np.array([1,2,3])
z=np.append(z,a)
for i in range(0,10):
    z=np.vstack([z,a])

z[9,2]=8
print(z)
print(len(z[:,2]))
print(z[len(z[:,2])-2,2])



