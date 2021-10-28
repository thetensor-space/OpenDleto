#module TensorSpace
using TensorOperations
A=randtensor((3,4,5),'Z')
IA=('s','b','c')
X1=randtensor((2,3),'Z')
Ix1('a','s')

C1=zeros(Int8,2,4,5)
IC1=('a','b','c')

tensorcontract!(1,A,IA,'N',X2,IX2,'N',0,C1,IC1)
##############################
B=randtensor((2,7,5),'Z')
IB('a','t','c')
Y1=randtensor((7,4),'Z')
IY1=('t','b')
C2=zeros(Int8,2,4,5)
IC2=('a','b','c')

tensorcontract!(1,B,IB,'N',Y1,IY1,'N',0,C2,IC2)
######################
C=C1+C2

X=ylvester(A,B,C)[1]

Y=sylvester(A,B,C)[2]



#end# End of module