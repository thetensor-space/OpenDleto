module TensorSpace
A=rand_tensor((3,4,5),'Z')

B=rand_tensor((3,4,5),'Z')

X1=rand_tensor((2,3),'Z')

Y1=rand_tensor((2,3),'Z')

C=contract(A,X1)+contract(B,Y1)

X=Sylvester(A,B,C)[1]

Y=Sylvester(A,B,C)[2]



end# End of module