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

Solucionar ecuaciones con variable compleja a través del método de Newton tiene una aplicación muy interesante en el campo
de los fractales como son las figuras fractales que se producen a partir de la convergencia,
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

Hágalo realidad ñero.

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
