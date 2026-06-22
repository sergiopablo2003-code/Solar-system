import numpy as np
import funciones as fc


c=1.49e11 #distancia de la tierra al sol en metros
G=6.67e-11 #constante de gravitacion universal
m=1.989e30 #masa del sol en kg
cv=np.sqrt(G*m/c)
M=np.array([1.0,0.33e24/m,4.87e24/m,5.97e24/m,0.642e24/m,1898e24/m,568e24/m,86.8e24/m,102e24/m], dtype=np.float64) #masas del sol y de mercurio
labels=["Sol", "Mercurio", "Venus", "Tierra", "Marte", "Jupiter", "Saturno", "Urano", "Neptuno" ] #etiquetamos los planetas
x=np.array([0.0,57.9e9/c,108.2e9/c,149.6e9/c,228e9/c,778.5e9/c,1432e9/c,2867e9/c,4515e9/c],dtype=np.float64)#posiciones en el eje x
y=np.zeros(len(x),dtype=np.float64)#posiciones en el eje y
b=np.zeros(len(x),dtype=np.float64)
a=x.copy()

vx=np.zeros(len(x),dtype=np.float64)
vy=np.array([0.0,47.4e3/cv,35.0e3/cv,29.8e3/cv,24.1e3/cv,13.1e3/cv,9.7e3/cv,6.8e3/cv,5.4e3/cv],dtype=np.float64)#velocidades en el eje y

f_x=np.zeros(len(x),dtype=np.float64)
f_y=np.zeros(len(x),dtype=np.float64)
f_x=fc.fuerza_x(x,y,M,np.zeros(len(x)))#inciializamos las fuerzas en x
f_y=fc.fuerza_y(x,y,M,np.zeros(len(x)))#inicializamos las fuerzas en y

w_x=np.zeros(len(x),dtype=np.float64)    
w_y=np.zeros(len(x),dtype=np.float64)

#Energía cinética, potencial y momento angular
T=np.zeros(len(x),dtype=np.float64)
V=np.zeros(len(x),dtype=np.float64)  
L=np.zeros(len(x),dtype=np.float64)
E=np.zeros(len(x),dtype=np.float64)
T=fc.Ec(vx,vy,M,np.zeros(len(x)))
V=fc.Ep(x,y,M,np.zeros(len(x)))
L=fc.momento_angular(x,y,vx,vy,M,np.zeros(len(x)))
E=T+V



t=0.0
h=0.01
t_c=np.sqrt(G*m/c**3)*t
tf=1000
#definimos las matrices y vectores en las que vamos a almacenar todos los datos que de nuestra simulación
P_x=np.empty([0,len(x)])
P_y=np.empty([0,len(x)])
Em=np.empty([0,len(x)])
Lm=np.empty([0,len(x)])
LTv= []
ETv= []

contadores=np.zeros(len(x))
tiempos=np.zeros(len(x))
periodos=np.zeros(len(x))



while t_c<tf:
      P_x=np.vstack([P_x,x])
      P_y=np.vstack([P_y,y])
      Em=np.vstack([Em,E])
      Lm=np.vstack([Lm,L])
      LT=fc.momento_angular_total(L)
      LTv.append(LT)
      ET= fc.Energía_total(E)
      ETv.append(ET)
      for k in range(0,len(x)):
         d=fc.distancia(x,y,k,0)
         w_x[k]=fc.w_nueva_x(vx,f_x,d,k,c,h)
         w_y[k]=fc.w_nueva_y(vy,f_y,d,k,c,h)
         x[k]=fc.posnueva_x(x,vx,f_x,d,k,c,h)
         y[k]=fc.posnueva_y(y,vy,f_y,d,k,c,h)
         f_x=fc.fuerza_x(x,y,M,f_x)
         f_y=fc.fuerza_y(x,y,M,f_y)
         vx[k]=fc.vel_nueva_x(w_x,f_x,d,k,c,h)
         vy[k]=fc.vel_nueva_y(w_y,f_y,d,k,c,h)
         L=fc.momento_angular(x,y,vx,vy,M,L)
         T=fc.Ec(vx,vy,M,T)
         V=fc.Ep(x,y,M,V)
         E=T+V

         # Si la coordenada y se vuelve negativa y aún no se ha registrado el tiempo, lo guardamos.
         if y[k] < 0 and P_y[len(P_y[:,k])-1,k]>0:
             contadores[k]=contadores[k]+1
             tiempos[k]=t_c
              

         

      t_c=t_c+h


print(contadores)
print (tiempos)

for i in range (1,5):
    periodos[i]=np.pi*(tiempos[i]/contadores[i])* 1.58e6 / 86400
    print(f"El periodo de {labels[i]} es: {periodos[i]:.2f} días")


for i in range (5,len(x)):
    periodos[i]=np.pi*8*(tiempos[i]/contadores[i])* 1.58e6 / 86400
    print(f"El periodo de {labels[i]} es: {periodos[i]:.2f} días")

