#module TensorSpace
A=randtensor((3,4,5),'Z')

B=randtensor((3,4,5),'Z')

X1=randtensor((2,3),'Z')

Y1=randtensor((2,3),'Z')

C=contract(A,X1)+contract(B,Y1)

X=ylvester(A,B,C)[1]

Y=sylvester(A,B,C)[2]



#end# End of module