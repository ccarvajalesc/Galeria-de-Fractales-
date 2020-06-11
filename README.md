<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>
  
# Fractales
## ¿Qué es un fractal?
Un fractal es un objeto cuya estructura se repite a diferentes escalas. Es decir, por mucho que nos acerquemos o alejemos del objeto, observaremos siempre la misma estructura. De hecho, somos incapaces de afirmar a qué distancia nos encontramos del objecto, ya que siempre lo veremos de la misma forma.

El termino fractal (del Latín fractus) fue propuesto por el matemático Benoît Mandelbrot en 1975. Los ejemplos más populares de fractales son el conjunto “Mandelbrot” o el triángulo “Sierpinski”. Este último se realiza de la siguiente manera: dibujamos un triángulo grande, colocamos otros tres triángulos en su interior a partir de sus esquinas, repetimos el último paso:

![sierspinki](https://raw.githubusercontent.com/ccarvajalesc/Galeria-de-Fractales-/master/Tri%C3%A1ngulo%20de%20Sierpinski.jpg)

## Fractales de Newton

Solucionar ecuaciones de variable compleja a través del método de Newton, tiene una aplicación muy interesante en el campo
de los fractales como lo son las figuras fractales que se producen a partir de la convergencia,
divergencia e incluso la eficiencia del método.

Como bien se sabe, el método de Newton se define de la siguiente forma:

$$x_{n+1}=x_n - \frac{f(x_n)}{f'(x_n)}$$

Sin embargo, el método de Newton para números complejos, se define así:

$$x_{n+1}=x_n - a\frac{f(x_n)}{f'(x_n)}$$

Donde $a$ es un número complejo distinto de cero, es decir, $a$ tiene la forma $a+bi$ y $bi=!0$.

Ya con lo anterior en mente, a continuación se presentan algunos fractales producidos por iteraciones del
método de Newton en el conjunto de los números complejos.

### Primer fractal de Newton:

Función utilizada: $z^{3}-1$

En este caso $a=\frac{1}{2}+\frac{i}{4}$

Fractal resultante:

![Newton1](https://raw.githubusercontent.com/ccarvajalesc/Galeria-de-Fractales-/master/Fractal%20Newton%201.png)

Código:

```
imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))
xa=-5
xb=5
ya=-5
yb=5
maxit=202
h=1e-6
eps=1e-3

def f(z):
    return z**3-1

for y in range (imgy):
    zy=y*(yb-ya)/(imgy-1)+ya
    for x in range (imgx):
        zx=x*(xb-xa)/(imgx-1)+xa
        z=complex(zx,zy)
        for i in range (maxit):
            dz=(f(z+complex(h,h))-f(z))/complex(h,h)     
            z0=z-(complex((1/2),(1/4))*(f(z)/dz))
            if abs (z0-z)<eps:
                break
            z=z0
            r=i*1
            g=i*12
            b=i*24
            image.putpixel((x,y),(r,g,b))
image
```
Específicamente, $a$ se encuentra en esta parte del código:

```
z0=z-(complex((1/2),(1/4))*(f(z)/dz))
```
### Segundo fractal de Newton:

Función utilizada: $z^{8}-15z^{4}-16$

Fractal resultante:

![Newton2](https://raw.githubusercontent.com/ccarvajalesc/Galeria-de-Fractales-/master/Fractal%20Newton%202.png)

Código:

```
xa=-5
xb=5
ya=-5
yb=5
maxit=202
h=1e-6
eps=1e-3

imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))

def f(z):
    return z**8+15*z**4-16

for y in range (imgy):
    zy=y*(yb-ya)/(imgy-1)+ya
    for x in range (imgx):
        zx=x*(xb-xa)/(imgx-1)+xa
        z=complex(zx,zy)
        for i in range (maxit):
            dz=(f(z+complex(h,h))-f(z))/complex(h,h)
            z0=z-f(z)/dz
            if abs (z0-z)<eps:
                break
            z=z0
            r=i*32
            g=i*7
            b=i*7
            image.putpixel((x,y),(r,g,b))
image
```
### Tercer fractal de Newton:

Función utilizada: $\frac{z^{3}}{sin(z)}$

En este caso $a=\frac{1}{2}+\frac{i}{4}$

Fractal resultante:

![Newton3](https://raw.githubusercontent.com/ccarvajalesc/Galeria-de-Fractales-/master/Fractal%20Newton%203.png)

Código:

```
xa=-5
xb=5
ya=-5
yb=5
maxit=100
h=1e-6
eps=1e-3

imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))

def f(z):
    return z**3/np.sin(z)

for y in range (imgy):
    zy=y*(yb-ya)/(imgy-1)+ya
    for x in range (imgx):
        zx=x*(xb-xa)/(imgx-1)+xa
        z=complex(zx,zy)
        for i in range (maxit):
            dz=(f(z+complex(h,h))-f(z))/complex(h,h)
            z0=z-(complex((1/2),(1/4))*(f(z)/dz))
            if abs (z0-z)<eps:
                break
            z=z0
            r=i*32
            g=i*8
            b=i*24
            image.putpixel((x,y),(r,g,b))
image
```
### Cuarto fractal de Newton:

Función utilizada: $tan(z^{3})-1+z^{3}$

Fractal resultante:

![Newton4](https://raw.githubusercontent.com/ccarvajalesc/Galeria-de-Fractales-/master/Fractal%20Newton%204.png)

Código:

```
xa=-5
xb=5
ya=-5
yb=5
maxit=20
h=1e-6
eps=1e-3

imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))

def f(z):
    return np.tan(z**3)-1+z**3

for y in range (imgy):
    zy=y*(yb-ya)/(imgy-1)+ya
    for x in range (imgx):
        zx=x*(xb-xa)/(imgx-1)+xa
        z=complex(zx,zy)
        for i in range (maxit):
            dz=(f(z+complex(h,h))-f(z))/complex(h,h)
            z0=z-f(z)/dz
            if abs (z0-z)<eps:
                break
            z=z0
            r=i*32
            g=i*15
            b=i*3
            image.putpixel((x,y),(r,g,b))
image
```
## Conjuntos de Julia:

Los conjuntos de Julia, así llamados por el matemático Gaston Julia, son una familia de conjuntos fractales que se obtienen al estudiar el comportamiento de los números complejos al ser iterados por una función.

Una familia muy importante de conjuntos de Julia se obtiene a partir de funciones cuadráticas simples,como por ejemplo:

$$F_c(z)=z^{2}+c$$
donde $c$ es un número complejo.

El conjunto de Julia que se obtiene a partir de esta función se denota $J_c$. El proceso para obtener este conjunto de Julia es el siguiente:

Se elige un número complejo cualquiera $z$ y se va construyendo una sucesión de números de la sigguiente manera:

$$z_0=z$$

$$z_1=F(z_0)=z_0^{2}+c$$

$$z_2=F(z_1)=z_1^{2}+c$$

$$\dots$$

$$z_{n+1}=F(z_n)=z_n^{2}+c$$

Si esta sucesión queda acotada, entonces se dice que $z$ pertenece al conjunto de Julia de parámetro $c$ denotado por $J_c$, de lo contrario, si la sucesión tiende a infinito $z$ queda excluído de éste.

Algunos ejemplos del conjunto de Julia se muestran a continuación:

### Conjunto de Julia 1:

Función utilizada: $z^{2}+(0+0.8i)$

Fractal resultante:

![Julia1](https://raw.githubusercontent.com/ccarvajalesc/Galeria-de-Fractales-/master/Conjunto%20Julia%205.png)

Código:

```
xa=-2
xb=2
ya=-2
yb=2
maxit=30

imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))

def f(z):
    return z**2+complex(0,0.8)

for y in range (imgy):
    zy=y*(yb-ya)/(imgy-1)+ya
    for x in range (imgx):
        zx=x*(xb-xa)/(imgx-1)+xa
        z=complex(zx,zy)
        for i in range (maxit):
            z0=f(z)
            if abs(z)>1000:
                break
            z=z0
            r=i*1
            g=i*8
            b=i*8
            image.putpixel((x,y),(r,g,b))
image
```
### Conjunto de Julia 2:

Función utilizada: $z^{7}-1+(cos(0.67)+\sqrt{0.5}i)$

Fractal resultante:

![Julia2](https://raw.githubusercontent.com/ccarvajalesc/Galeria-de-Fractales-/master/Conjunto%20Julia%202.png)

Código:

```
xa=-2
xb=2
ya=-2
yb=2
maxit=30

imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))

def f(z):
    return z**7-1+complex(np.cos(0.67),np.sqrt(0.5))

for y in range (imgy):
    zy=y*(yb-ya)/(imgy-1)+ya
    for x in range (imgx):
        zx=x*(xb-xa)/(imgx-1)+xa
        z=complex(zx,zy)
        for i in range (maxit):
            z0=f(z)
            if abs(z)>1000:
                break
            z=z0
            r=i*32
            g=i*13
            b=i*20
            image.putpixel((x,y),(r,g,b))
image
```
### Conjunto de Julia 3:

Función utilizada: $(\frac{1}{z}+z^{2}i)+(0+0.9i)$

Fractal resultante:

![Julia3](https://raw.githubusercontent.com/ccarvajalesc/Galeria-de-Fractales-/master/Conjunto%20Julia%203.png)

Código:

```
xa=-2
xb=2
ya=-2
yb=2
maxit=30

imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))

def f(z):
    return complex(1/z,z**2)+complex(0,0.9)

for y in range (imgy):
    zy=y*(yb-ya)/(imgy-1)+ya
    for x in range (imgx):
        zx=x*(xb-xa)/(imgx-1)+xa
        z=complex(zx,zy)
        for i in range (maxit):
            z0=f(z)
            if abs(z)>1000:
                break
            z=z0
            r=i*6
            g=i*1
            b=i*8
            image.putpixel((x,y),(r,g,b))
image
```
### Conjunto de Julia 4:

Función utilizada: $(3z^{5}+\frac{z^{3}}{z^{2}-1}i)+(0+0.9i)$

Fractal resultante:

![Julia4](https://raw.githubusercontent.com/ccarvajalesc/Galeria-de-Fractales-/master/Conjunto%20Julia%204.png)

Código:

```
xa=-2
xb=2
ya=-2
yb=2
maxit=30

imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))

def f(z):
    return complex(3*z**5,z**3/z**2-1)+complex(0,0.9)

for y in range (imgy):
    zy=y*(yb-ya)/(imgy-1)+ya
    for x in range (imgx):
        zx=x*(xb-xa)/(imgx-1)+xa
        z=complex(zx,zy)
        for i in range (maxit):
            z0=f(z)
            if abs(z)>1000:
                break
            z=z0
            r=i*10
            g=i*16
            b=i*32
            image.putpixel((x,y),(r,g,b))
image
```
## Fractales de Sistemas Iterados de Funciones (IFS):

Los sistemas de funciones iteradas (SFI por sus siglas en inglés) son una herramienta matemática sencilla para construir conjuntos fractales por medio de un conjunto de aplicaciones afines contractivas. Este método fue desarrollado por M.F. Barnsley en 1985. En concreto, resulta de utilidad para obtener un fractal autosemejante a base de aplicar de forma iterativa el sistema de funciones a un conjunto cualquiera, hasta llegar a una buena aproximación del fractal que constituye el atractor del sistema.

Los sistemas de funciones iteradas son conjuntos de n transformaciones afines contractivas. Normalmente se utilizan dos tipos de algoritmos, el algoritmo determinista y el algoritmo aleatorio.

### Algoritmo determinista:

El algoritmo determinista consiste en tomar un conjunto de puntos, que puede ser cualquier figura geométrica, y aplicarle cada una de las n transformaciones afines del sistema, con lo cual obtenemos n conjuntos de puntos transformados. A cada uno de ellos le volvemos a aplicar cada una de las n funciones, obteniendo n2 nuevos conjuntos de puntos.

Continuamos de esta manera iterando sobre los resultados, hasta que la unión de todos los conjuntos obtenidos en la última iteración se va aproximando a la figura que constituye el atractor del sistema. A este atractor llegaremos siempre, independientemente de la forma conjunto de partida. Cada IFS tiene un atractor característico, que será un fractal autosemejante, ya que está construido a base de copias de sí mismo, cada vez más pequeñas. Normalmente no hacen falta muchas iteraciones para obtener dicho conjunto fractal.

### Algoritmo aleatorio:

El algoritmo aleatorio es similar, pero en lugar de aplicar las funciones a un conjunto de puntos, las aplicamos sobre un único punto, que vamos dibujando. A cada una de las transformaciones del sistema le asignamos un valor de probabilidad, teniendo en cuenta que la suma total de los valores de probabilidad de las funciones debe valer 1. En cada iteración del algoritmo, seleccionamos una de las transformaciones con probabilidad p. Esto es muy sencillo de hacer, simplemente se obtiene un valor aleatorio entre 0 y 1, por ejemplo con la clase Random, y se van sumando una por una las probabilidades de cada función, hasta que el resultado sea mayor que el número aleatorio obtenido. Esa será la función seleccionada.

Los primeros puntos de la serie se descartan. Porque normalmente están muy alejados del atractor, el resto se van dibujando hasta obtener el dibujo del fractal correspondiente, normalmente después de un número de iteraciones entre 1000 y 5000.

### Fractales generados por un algoritmo determinista:

#### Primer fractal generado por un algoritmo determinista:

*Curva del dragón*

![CurvadelDragón](https://raw.githubusercontent.com/ccarvajalesc/Galeria-de-Fractales-/master/Curva%20del%20Drag%C3%B3n.png)

Código:

```
import matplotlib.pyplot as plt
import argparse
import math

def Dragon(level, initial_state, trgt, rplcmnt, trgt2, rplcmnt2):
    state = initial_state
   
    for counter in range(level):
        state2 = ''
        for character in state:
            if character == trgt:
                state2 += rplcmnt
            elif character == trgt2:
                state2 += rplcmnt2
            else:
                state2 += character
        state = state2
    return state

totalwidth=100
iterations = 20

plt.figure(figsize=(15,15))

points = dragon(iterations,totalwidth,(-totalwidth/2,0))
plt.plot([p[0] for p in points], [p[1] for p in points], '-',color='darkviolet')
plt.axis('equal')

plt.title("Curva del Dragón. Iteración = 20")

plt.show()
```
#### Segundo fractal generado por un algoritmo determinista:

*Triángulo de Sierpinski*

![SierpinskiTriangle](https://raw.githubusercontent.com/ccarvajalesc/Galeria-de-Fractales-/master/Triangle%20Sierpinski.png)

Código:

```
fig=plt.figure()
ax=plt.gca()
Tri=np.array([[0,0],[1,0],[0,1],[0,0]])

def transafin(M,t,x):
    y=M@x+t
    return y

transafin([[0.5,0],[0,0.5]],[0,0],Tri[1])

Tri=np.array([[0,0],[1,0],[0,1],[0,0]])
tritrans=np.array([transafin([[0.5,0],[0,0.5]],[0,0],i) for i in Tri])
tritrans2=np.array([transafin([[0.5,0],[0,0.5]],[0,0.5],i) for i in Tri])
tritrans3=np.array([transafin([[0.5,0],[0,0.5]],[0.5,0],i) for i in Tri])

Tri=np.concatenate((tritrans,tritrans2,tritrans3))

Tri=np.array([[0,0]])
for i in range(8):
    tritrans=np.array([transafin([[0.5,0],[0,0.5]],[0,0],i) for i in Tri])
    tritrans2=np.array([transafin([[0.5,0],[0,0.5]],[0,0.5],i) for i in Tri])
    tritrans3=np.array([transafin([[0.5,0],[0,0.5]],[0.5,0],i) for i in Tri])
    Tri=np.concatenate((tritrans,tritrans2,tritrans3))
plt.scatter(Tri.transpose()[0],Tri.transpose()[1],color='aqua',s=0.2)
ax.set_xticks(np.arange(-0.2,1.4,0.2))
ax.set_yticks(np.arange(-0.2,1.4,0.2))
plt.grid()
ax.axis("equal")
```
### Fractales generados por un algoritmo aleatorio:

#### Primer fractal generado por un algoritmo aleatorio:

*Árbol binario:*

![Árbolbinario](https://raw.githubusercontent.com/ccarvajalesc/Galeria-de-Fractales-/master/%C3%81rbol%20binario.png)

Código:

```
import matplotlib.pyplot as plt
import math


def dibujaTree(x1, y1, angle, depth):

    if depth:
        x2 = x1 + int(math.cos(math.radians(angle)) * depth * 10.0)
        y2 = y1 + int(math.sin(math.radians(angle)) * depth * 10.0)
        plt.plot([x1,x2],[y1,y2],'-',color='darkgreen',lw=3)
        dibujaTree(x2, y2, angle - 20, depth - 1)
        dibujaTree(x2, y2, angle + 20, depth - 1)


plt.figure(figsize=(10,10))
depth = 10
dibujaTree(100, 350, 90, depth)

plt.show()
```
### Segundo fractal generado por un algoritmo aleatorio:

*Curva de Hilbert:*

![CurvadeHilbert](https://raw.githubusercontent.com/ccarvajalesc/Galeria-de-Fractales-/master/Hilbert.png)

Código:

```
import sys, math
import numpy as np
import matplotlib.pyplot as plt
  
def hilbert(x0, y0, xi, xj, yi, yj, n,points):
    if n <= 0:
        X = x0 + (xi + yi)/2
        Y = y0 + (xj + yj)/2
        points.append((X,Y))
    else:
        hilbert(x0,               y0,               yi/2, yj/2, xi/2, xj/2, n - 1,points)
        hilbert(x0 + xi/2,        y0 + xj/2,        xi/2, xj/2, yi/2, yj/2, n - 1,points)
        hilbert(x0 + xi/2 + yi/2, y0 + xj/2 + yj/2, xi/2, xj/2, yi/2, yj/2, n - 1,points)
        hilbert(x0 + xi/2 + yi,   y0 + xj/2 + yj,  -yi/2,-yj/2,-xi/2,-xj/2, n - 1,points)
        return points

        
a = np.array([0, 0])
b = np.array([1, 0])
c = np.array([1, 1])
d = np.array([0, 1])

iterations = 5

plt.subplot(1,2,2).set_title("Iteración = 5")

points = hilbert(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, iterations,[])
plt.plot([p[0] for p in points], [p[1] for p in points], '-',lw=3,color='orange')#,lw=5)

plt.axis('equal')
```

## Fractales en 3D:

A continuación se muestran un par de fractales de Newton en tres dimensiones:

### Fractal 3D I:

![newton3D1](https://raw.githubusercontent.com/ccarvajalesc/Galeria-de-Fractales-/master/Fractal%203D%201.png)

Código:

```
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.figure as fg
from matplotlib import cm 
import numpy as np 

fig = plt.figure() 
ax = fig.add_subplot(111, projection='3d')
ax.view_init(azim=-130,elev=45) 
ax.dist = 4.3 
ax.set_facecolor([.85,.85,.45]) 

n = 8 
dx = 0.0 
dy = 0.0 
L = 1.0 
M = 300 

def f(Z): 
    return np.e**(-np.abs(Z))

x = np.linspace(-L+dx,L+dx,M) 
y = np.linspace(-L+dy,L+dy,M) 
X,Y = np.meshgrid(x,y) 
Z = X + 1j*Y 

for k in range(1,n+1): 
    ZZ = Z - (Z**4 + 1)/(4*Z**3)
    Z = ZZ
    W = f(Z)
    
ax.set_xlim(dx-L,dx+L) 
ax.set_zlim(dy-L,dy+L) 
ax.set_zlim(-2.5*L,2*L) 
ax.axis("off") 
ax.plot_surface(X, Y, -W, rstride=1, cstride=1, cmap="ocean") 
plt.show() 
```

### Fractal 3d II:

![newton3D2](https://raw.githubusercontent.com/ccarvajalesc/Galeria-de-Fractales-/master/Fractal%203D%202.png)

Código:

```
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.figure as fg
from matplotlib import cm 
import numpy as np 

fig = plt.figure() 
ax = fig.add_subplot(111, projection='3d')
ax.view_init(azim=-140,elev=45) 
ax.dist = 3.0 
ax.set_facecolor([1.0,.15,.15]) 

n = 8 
dx = 0.0 
dy = 0.0 
L = 1.3 
M = 300 

def f(Z):  
    return np.e**(-np.abs(Z))

x = np.linspace(-L+dx,L+dx,M) 
y = np.linspace(-L+dy,L+dy,M) 
X,Y = np.meshgrid(x,y) 
Z = X + 1j*Y 

for k in range(1,n+1): 
    ZZ = Z - (Z**4 + 1)/(4*Z**3)
    Z = ZZ
    W = f(Z)
    
ax.set_xlim(dx-L,dx+L) 
ax.set_zlim(dy-L,dy+L) 
ax.set_zlim(-3.5*L,4*L) 
ax.axis("off") 
ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap='flag') 
plt.show()
```

## Bibliografía:

- gobiernodecanarias (s.f). CONJUNTOS DE JULIA Y MANDELBROT. Recuperado el 8 de junio de 2020, disponible en: http://www3.gobiernodecanarias.org/medusa/ecoblog/mrodperv/fractales/conjuntos-de-julia-y-mandelbrot/

- CSCAZORLA (2012). ¿Qué son los fractales y cómo se construyen?. Recuperado el 8 de junio de 2020, disponible en: https://www.xatakaciencia.com/matematicas/que-son-los-fractales-y-como-se-construyen

- Díaz, M (2017). Dibujar fractales con Sistemas de Funciones Iteradas (IFS). Recuperado el 8 de junio de 2020, disponible en: http://software-tecnico-libre.es/es/articulo-por-tema/todas-las-secciones/todos-los-temas/todos-los-articulos/dibujar-con-sistemas-de-funciones-iteradas
