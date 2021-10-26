function flatten(A,T) #data of A is of type T, A is an IxJxK tensor 
s=size(A)
pA=prod(s) #gives product of the dimensions of A
flatA=zeros(T,pA)
for i=1:s[1]
    for j=1:s[2]
        for k=1:s[3]
            eqindex= i+s[1]*((j-1)+s[2]*(k-1))
            flatA[eqindex] = A[i,j,k]
        end
    end
end
return flatA
end #end of flatten