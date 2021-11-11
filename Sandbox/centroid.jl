using LinearAlgebra
#solving h(u*v)=f(u)*v=u*g(v)
#ΓZ=XΓ=ΓY^T
#Where
#[ΓZ]_{ijk}=ΣΓ_{ijl}Z_{lk} = ΣX_{il}Γ_{ljk} = ΣΓ_{ilk}Y_{jl}
function cent(L)
    I=size(L)[1]#X
    J=size(L)[2]#Y
    K=size(L)[3]#Z
    M=zeros(I*J*K,I^2+J^2+K^2)
    for i=1:I
        for j=1:J
            index=i+I*(j-1)
            M[index:index+I*J*(K-1),i*I:(i+1)*I]= L[1:I,j,1:K]#X  
            M[index:index+I*J*(K-1),I^2+j*J:(j+1)*I]=L[i,1:J,1:K]#Y
        end
    end
    return M
end