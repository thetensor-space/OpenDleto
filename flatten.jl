function flatten(A,T) #data of A is of type T 
s=size(A)
pA=prod(s) #gives product of the dimensions of A
flatA=zeros(T,pA)
for i=1:length(s)
    y=s[i]
    for j=1:i
        flatA[i+a*(j+b*k)] = A[]
    end
end #end of flatten