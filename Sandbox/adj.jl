using LinearAlgebra
#solving f(u)*v=u*g(v)
#XΓ=ΓY^T
#Where
#ΣX_{il}Γ_{ljk} = ΣΓ_{ilk}Y_{jl}
function adj(L)
    I=size(L)[1]#X
    J=size(L)[2]#Y
    K=size(L)[3]#Z
    M=zeros(I*J*K,I^2+J^2+K^2)
    for i=1:I
        for j=1:J
            for k=1:K
                index=i+I*(j-1) +I*J*(k-1)
                M[index,i*I:(i+1)*I-1]= L[1:I,j,k]#X  
                M[index,I^2+j*I:I^2+(j+1)*I-1]=L[i,1:J,k]#Y
            end
        end
    end
    return M
end