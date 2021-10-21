#module TensorSpace

using LinearAlgebra

function contract(A,B)
sA=size(A)# A is (I,J,K)
I=sA[1]
J=sA[2]
K=sA[3]

sB=size(B)# B is (L,I)
L=sB[1]

#Here I need to make sure dimensions match up correctly
T=Float64 #Type for the ring
C=zeros(T, L,J,K)

for l= 1:L
    for j= 1:J
        for k=1:K
            vA=A[:,j,k]
            vB=B[l,:]
            dotAB=dot(vA,vB) 
            C[l,j,k]= dotAB
        end#K
    end#J         
end#L
return C
end#End of contract

#end #End of module