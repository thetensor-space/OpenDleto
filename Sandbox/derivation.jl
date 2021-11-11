using LinearAlgebra
#solving d(u*v)=d(u)*v+u*d(v)
#ΓZ=XΓ+ΓY^T
#Where
#[ΓZ]_{ijk}=ΣΓ_{ijl}Z_{lk} = ΣZ_{il}Γ_{ljk} + ΣΓ_{ilk}Y_{jl} 
function der(L)
    I=size(L)[1]#X
    J=size(L)[2]#Y
    K=size(L)[3]#Z
    M=zeros(I*J*K,I^2+J^2+K^2)
    for i=1:I
        for j=1:J
            for k=1:K
                index=i+I*((j-1)+J*(k-1))
                M[index,i*I:(i+1)*I]= L[1:I,j,k] #X
                M[index,I^2+j*J:(j+1)*I]=L[i,1:J,k]#Y
                M[index,I^2+J^2+k*K:(k+1)*K]=L[i,j,1:K].*(-1)#Z^T, permuting columns for faster runtime
            end
        end
    end
    return M
end