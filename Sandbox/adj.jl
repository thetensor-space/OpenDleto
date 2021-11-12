using LinearAlgebra
#solving f(u)*v=u*g(v)
#XΓ=ΓY^T
#Where
#ΣX_{il}Γ_{ljk} = ΣΓ_{ilk}Y_{jl}

function adjRow(t, i,j,k)
    F = eltype(t)
    a = size(t)[1]; b = size(t)[2]
    row = zeros(F,1,a^2+b^2)
    copyto!(row, a*(i-1)+1, t[1:a,j,k], 1)
    copyto!(row, a^2+b*(j-1)+1, -t[i,1:b,k], 1)
    return row
end

function adj(t)
    F = eltype(t)
    a = size(t)[1]; b = size(t)[2]; c = size(t)[3]
    M = zeros(F, (0,a^2+b^2) )
    for k = axes(t,3)
        for j = axes(t,2) 
            for i = axes(t,1)
                M = vcat(M,adjRow(t, i,j,k))
            end
        end
    end
    return M
end 

# function adj(L)
#     I=size(L)[1]#X
#     J=size(L)[2]#Y
#     K=size(L)[3]#Z
#     M=zeros(I*J*K,I^2+J^2+K^2)
#     for i=1:I
#         for j=1:J
#             for k=1:K
#                 index=i+I*(j-1) +I*J*(k-1)
#                 M[index,i*I:(i+1)*I-1]= L[1:I,j,k]#X  
#                 M[index,I^2+j*I:I^2+(j+1)*I-1]=L[i,1:J,k]#Y
#             end
#         end
#     end
#     return M
# end

#############################################
## Examples.
#############################################

## Flat Genus 2 Indecomposable
function FlatGenus2(F,d)
    t = zeros(F, (2*d+1,2*d+1,2))
    for i = 1:d
        t[i,d+i,1] = 1;   t[d+i,i,1] = -1
        t[i,d+i+1,2] = 1; t[d+i+1,i,2] = -1
    end
    return t
end 


