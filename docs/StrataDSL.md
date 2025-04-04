# Strata DSL

### Loading tensors
```
t = as tensor "data.csv" [axis_1] [axis_2] ... [axis_l] value
```

E.g.
```
t = as tensor "games.csv" [Action,Strategy,Adventure] [PSP,Wii,XB] [Score] Global_Sales
```

```
t = as tensor [2,3,4] 1 1 0; 0 1 1;; 0 1 1; 0 0 1;; 0 0 1; 0 0 0
```

```
surface equation(x,y,z) [xrange] [yrange] [zrange]
t = as tensor support
```

```
t = as tensor image surface
```

```
add noise [distribution] [params]
```

```
randomize t [nonorthogonal]
```


### Displaying tensors

```
print t
plot t
slideshow t axis
```

### Storing tensors

```
save t file
as csv t
```

### Chisels

```
nondegenerate t [axes]
tucker t [axes]
```

```
s = stratify t
b = blocks t [axes] (max_blocks)
```
Use decay rates of singular values to separate blocks by kmeans.

```
so = stratify t nonorthogonal
```



List all chisels (with parameters if infinite)
```
chisels valence
```




### Experiments

