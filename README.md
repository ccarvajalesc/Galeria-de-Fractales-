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

Donde $a$ es un número complejo distinto de cero, es decir, $a$ tiene la forma $a+bi$ y $bi=!0$

Ya con lo anterior en mente, a continuación se presentan algunos fractales producidos por iteraciones del
método de Newton en el conjunto de los números complejos.

### Primer fractal de Newton:

Función utilizada: $z^{3}-1$

En este caso $a=\frac{1}{2}+\frac{i}{4}$

Fractal resultante:

![newton1](https://raw.githubusercontent.com/ccarvajalesc/Galeria-de-Fractales-/master/Fractal%20Newton%201.png)

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

