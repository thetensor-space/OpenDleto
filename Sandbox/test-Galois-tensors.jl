using TensorOperations 
#XA
A=galoisrandtensor((3,4,5),3,3)
IA=('s','b','c')
X1=galoisrandtensor((2,3),3,3)
IX1=('a','s')

C1=galoisrandtensor((2,4,5),3,3)
IC1=('a','b','c')

tensorcontract!(1,A,IA,'N',X1,IX1,'N',0,C1,IC1)



#BY
B=galoisrandtensor((2,7,5),3,3)
IB=('a','t','c')
Y1=galoisrandtensor((7,4),3,3)
IY1=('t','b')
C2=galoisrandtensor((2,4,5),3,3)
IC2=('a','b','c')

tensorcontract!(1,B,IB,'N',Y1,IY1,'N',0,C2,IC2)
######################
C=C1+C2


#X=sylvestergalois(A,B,C,Int8)[1]

#Y=sylvestergalois(A,B,C,Int8)[2]