#module TensorSpace
using TensorOperations #To use tensorcontract
T='Q' #Current field
#XA
A=randtensor((3,4,5),T)
IA=('s','b','c')
X1=randtensor((2,3),T)
IX1=('a','s')

C1=zeros(Int8,2,4,5).//1
IC1=('a','b','c')

tensorcontract!(1,A,IA,'N',X1,IX1,'N',0,C1,IC1)

#BY
B=randtensor((2,7,5),T)
IB=('a','t','c')
Y1=randtensor((7,4),T)
IY1=('t','b')
C2=zeros(Int8,2,4,5).//1
IC2=('a','b','c')

tensorcontract!(1,B,IB,'N',Y1,IY1,'N',0,C2,IC2)
######################
C=C1+C2
#return C
return sylvester(A,B,C,Int8)
#X=sylvester(A,B,C,Int8)[1]

#Y=sylvester(A,B,C,Int8)[2]



#end# End of module