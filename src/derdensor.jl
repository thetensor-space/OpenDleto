## Valence 3 tensor
function sylvester3(ch::Matrix,t::AbstractArray)
    dims = size(t)
    C1 = ch[:,1]
    C2 = ch[:,2]
    C3 = ch[:,3]
    function ester(x) 
        # X = transverse(ops, x)
        offset = 0; 
        X1 = reshape(view(x[1:(dims[1]*dims[1])]), (dims[1], dims[1]))
        offset += dims[1]*dims[1]
        X2 = reshape(view(x[(offset+1):(offset + dims[2]*dims[2])]), (dims[2], dims[2]))
        offset += dims[2]*dims[2]
        X3 = reshape(view(x[(offset+1):(offset + dims[3]*dims[3])]), (dims[3], dims[3]))
        # Chisels are tiny, no point in using views.
        @tensor Y[c,i,j,k] = C1[c]*t[i',j,k]*X1[i,i'] + C2[c]*t[i,j',k]*X2[j,j'] + C3[c]*t[i,j,k']*X3[k,k']
        return vec(Y)
    end
    function sylv(y)
        Y = reshape(y, (size(ch,1), size(t,1), size(t,2), size(t,3)))
        @tensor X1[i,i'] = t[i,j,k] * C1[c] * Y[c,i',j,k]
        @tensor X2[j,j'] = t[i,j,k] * C2[c] * Y[c,i,j',k] 
        @tensor X3[k,k'] = t[i,j,k] * C3[c] * Y[c,i,j,k']
        return [vec(M) for M in mats] |> vcat
        # return unsafe_contains(ops, collect(X1,X2,X3))
    end

    # Compose sylv  ester
    function sylvester(x)
        # X = transverse(ops, x)
        offset = 0; 
        X1 = reshape(view(x,1:(dims[1]*dims[1])), (dims[1], dims[1]))
        offset += dims[1]*dims[1]
        X2 = reshape(view(x,(offset+1):(offset + dims[2]*dims[2])), (dims[2], dims[2]))
        offset += dims[2]*dims[2]
        X3 = reshape(view(x,(offset+1):(offset + dims[3]*dims[3])), (dims[3], dims[3]))
        Z1 = similar(X1); Z2 = similar(X2); Z3 = similar(X3);
        @tensor Z1[i,i'] = t[i,j,k]*C1[c]*C1[c]*t[i'',j,k]*X1[i',i''] + t[i,j,k]*C1[c]*C2[c]*t[i',j',k]*X2[j,j'] + t[i,j,k]*C1[c]*C3[c]*t[i',j,k']*X3[k,k']
        @tensor Z2[j,j'] = t[i,j,k]*C2[c]*C1[c]*t[i',j',k]*X1[i,i'] + t[i,j,k]*C2[c]*C2[c]*t[i,j'',k]*X2[j',j''] + t[i,j,k]*C2[c]*C3[c]*t[i,j',k']*X3[k,k']
        @tensor Z3[k,k'] = t[i,j,k]*C3[c]*C1[c]*t[i',j,k']*X1[i,i'] + t[i,j,k]*C3[c]*C2[c]*t[i,j',k']*X2[j,j'] + t[i,j,k]*C3[c]*C3[c]*t[i,j,k'']*X3[k',k'']
        return vcat(vec(Z1), vec(Z2), vec(Z3))
    end
    #TBD: for convergence we should work on conditioning ch' ch,
    # something like a Grassmannian Frames (tight frames)--see Emily King.

    e = LinearMap( ester, sylv, size(t,1)^2+size(t,2)^2+size(t,3)^2, size(ch,1)*size(t,1); ismutating=false )
    se = LinearMap( sylvester,  size(t,1)^2+size(t,2)^2+size(t,3)^2, size(t,1)^2+size(t,2)^2+size(t,3)^2; ismutating=false, issymmetric=true, isposdef=false )
    return se, e
end


## Valence 3 tensor
function derdensor3(ch::Matrix,t::AbstractArray)
    dims = size(t)
    C1 = ch[:,1]
    C2 = ch[:,2]
    C3 = ch[:,3]
    function densor(x) 
        # X = transverse(ops, x)
        offset = 0; 
        X1 = reshape(view(x[1:(dims[1]*dims[1])]), (dims[1], dims[1]))
        offset += dims[1]*dims[1]
        X2 = reshape(view(x[(offset+1):(offset + dims[2]*dims[2])]), (dims[2], dims[2]))
        offset += dims[2]*dims[2]
        X3 = reshape(view(x[(offset+1):(offset + dims[3]*dims[3])]), (dims[3], dims[3]))
        # Chisels are tiny, no point in using views.
        Y = zeros( size(ch,1), dims[1], dims[2], dims[3] )
        @tensor Y[c,i,j,k] = C1[c]*t[i',j,k]*X1[i,i'] + C2[c]*t[i,j',k]*X2[j,j'] + C3[c]*t[i,j,k']*X3[k,k']
        return vec(Y)
    end
    function der(y)
        Y = reshape(y, (size(ch,1), size(t,1), size(t,2), size(t,3)))
        @tensor X1[i,i'] = t[i,j,k] * C1[c] * Y[c,i',j,k]
        @tensor X2[j,j'] = t[i,j,k] * C2[c] * Y[c,i,j',k] 
        @tensor X3[k,k'] = t[i,j,k] * C3[c] * Y[c,i,j,k']
        return [vec(M) for M in mats] |> vcat
        # return unsafe_contains(ops, collect(X1,X2,X3))
    end

    # Compose sylv  ester
    function derdensor(x)
        # X = transverse(ops, x)
        offset = 0; 
        X1 = reshape(view(x,1:(dims[1]*dims[1])), (dims[1], dims[1]))
        offset += dims[1]*dims[1]
        X2 = reshape(view(x,(offset+1):(offset + dims[2]*dims[2])), (dims[2], dims[2]))
        offset += dims[2]*dims[2]
        X3 = reshape(view(x,(offset+1):(offset + dims[3]*dims[3])), (dims[3], dims[3]))
        Z1 = similar(X1); Z2 = similar(X2); Z3 = similar(X3);
        Y = zeros( size(ch,1), dims[1], dims[2], dims[3] )
        @tensor Y[c,i,j,k] = C1[c]*t[i',j,k]*X1[i,i'] + C2[c]*t[i,j',k]*X2[j,j'] + C3[c]*t[i,j,k']*X3[k,k']
        @tensor Z1[i,i'] = t[i,j,k]*C1[c]*Y[c,i',j,k]
        @tensor Z2[j,j'] = t[i,j,k]*C2[c]*Y[c,i,j',k]
        @tensor Z3[k,k'] = t[i,j,k]*C3[c]*Y[c,i,j,k']
        return vcat(vec(Z1), vec(Z2), vec(Z3))
    end
    #TBD: for convergence we should work on conditioning ch' ch,
    # something like a Grassmannian Frames (tight frames)--see Emily King.

    e = LinearMap( densor, der, size(t,1)^2+size(t,2)^2+size(t,3)^2, size(ch,1)*size(t,1); ismutating=false )
    se = LinearMap( derdensor,  size(t,1)^2+size(t,2)^2+size(t,3)^2, size(t,1)^2+size(t,2)^2+size(t,3)^2; ismutating=false, issymmetric=true, isposdef=false )
    return se, e
end
