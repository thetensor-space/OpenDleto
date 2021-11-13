########################################################################
#  CC-BY 2021  Amaury V. Miniño and James B. Wilson.
#    Distributed under MIT License.
########################################################################

using LinearAlgebra


#solving d(u*v)=d(u)*v+u*d(v)
#ΓZ=XΓ+ΓY^T
#Where
#[ΓZ]_{ijk}=ΣΓ_{ijl}Z_{lk} = ΣZ_{il}Γ_{ljk} + ΣΓ_{ilk}Y_{jl} 


function derRow(t, i,j,k)
    F = eltype(t)
    a = size(t)[1]; b = size(t)[2]; c = size(t)[3]
    row = zeros(F,1,a^2+b^2+c^2)
    # ΣZ_{il}Γ_{ljk}
    copyto!(row, a*(i-1)+1, t[1:a,j,k], 1)
    # ΣΓ_{ilk}Y_{jl}
    copyto!(row, a^2+b*(j-1)+1, -t[i,1:b,k], 1)
    # -ΣΓ_{ijl}Z_{lk}
    copyto!(row, a^2+b^2+c*(k-1)+1, -t[i,j,1:c], 1)
    return row
end

function der(t)
    F = eltype(t)
    a = size(t)[1]; b = size(t)[2]; c = size(t)[3]
    step = a^2+b^2+c^2;
    M = zeros(F, a*b*c*step )
    for i = axes(t,1)
        for j = axes(t,2) 
            for k = axes(t,3)
                row = i-1+a*((j-1)+b*(k-1))
                # ΣZ_{il}Γ_{ljk}
                copyto!(M, step*row+a*(i-1)+1, t[1:a,j,k], 1)
                # ΣΓ_{ilk}Y_{jl}
                copyto!(M, step*row+a^2+b*(j-1)+1, -t[i,1:b,k], 1)
                # -ΣΓ_{ijl}Z_{lk}
                copyto!(M, step*row+ a^2+b^2+c*(k-1)+1, -t[i,j,1:c], 1)
                #M = vcat(M,derRow(t, i,j,k))
            end
        end
    end
    return reshape( M, (step,a*b*c) )
end 
