using JuAFEM, SparseArrays
# JuAFEM needs to be installed....


function doassemble_K!(K::SparseMatrixCSC, f::Vector, cellvalues::CellScalarValues{dim}, dh::DofHandler) where {dim}

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    assembler = start_assemble(K, f)

    @inbounds for cell in CellIterator(dh)

        fill!(Ke, 0)
        fill!(fe, 0)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dOmega = getdetJdV(cellvalues, q_point)

            for i in 1:n_basefuncs
                v  = shape_value(cellvalues, q_point, i)
                gv = shape_gradient(cellvalues, q_point, i)
                fe[i] += v * dOmega
                for j in 1:n_basefuncs
                    gu = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (gv ⋅ gu) * dOmega
                end
            end
        end

        assemble!(assembler, celldofs(cell), fe, Ke)
    end
    return K, f
end

function doassemble_M!(M::SparseMatrixCSC, cellvalues::CellScalarValues{dim}, dh::DofHandler) where {dim}

    n_basefuncs = getnbasefunctions(cellvalues)
    Me = zeros(n_basefuncs, n_basefuncs)

    assembler = start_assemble(M)

    @inbounds for cell in CellIterator(dh)

        fill!(Me, 0)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dOmega = getdetJdV(cellvalues, q_point)

            for i in 1:n_basefuncs
                v  = shape_value(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    u = shape_value(cellvalues, q_point, j)
                    Me[i, j] += (v ⋅ u) * dOmega
                end
            end
        end

        assemble!(assembler, celldofs(cell), Me)
    end
    return M
end

# the vector is arranged in a strange way need to be twisted
function twistMatrix( m )
	xsize = size(m)[1] 
	ysize = size(m)[2] 
	res = copy(m)
	firstcols = reshape(m[1:xsize, 1:2],(2,xsize))'
	for i=1:xsize
		res[i,1] = firstcols[i,1]
		res[i,2] = firstcols[i,2]
	end
	res[2,1] = firstcols[2,2]
	res[2,2] = firstcols[2,1]
	firstrows = res[1:2,1:ysize]
	for j=3:ysize
		res[1,j] = firstrows[2,j]
		res[2,j] = firstrows[1,j]
	end
	return res
end

function untwistMatrix( m )
	xsize = size(m)[1] 
	ysize = size(m)[2] 
	res = copy(m)
	firstcols = reshape(m[1:xsize, 1:2]',(xsize,2))
	for i=1:xsize
		res[i,1] = firstcols[i,1]
		res[i,2] = firstcols[i,2]
	end
	res[3,1] = firstcols[4,1]
	res[4,1] = firstcols[3,1]
	firstrows = res[1:2,1:ysize]
	for j=3:ysize
		res[1,j] = firstrows[2,j]
		res[2,j] = firstrows[1,j]
	end
	return res
end



function solveHeat(initial,dt,tmax)
	# make a tensor for the output
	xsize = size(initial)[1] 
	ysize = size(initial)[2]
	tsize = trunc(Int, tmax/dt) + 1

	output = zeros(Float32, xsize*ysize*tsize )
	extinititalflat = zeros(Float32,xsize*ysize);
	for j=2:(ysize-1)
		copyto!(extinititalflat, (j-1)*xsize + 2, initial[2:xsize-1,j], 1)
	end
	copyto!(output,1,extinititalflat,1)
#	display(reshape(extinititalflat,(xsize,ysize)))
	
	# set initial conditions
	un = reshape(untwistMatrix(reshape(extinititalflat,(xsize,ysize))), xsize*ysize)

	grid = generate_grid(Quadrilateral, (xsize-1, ysize-1));

	dim = 2
	ip = Lagrange{dim, RefCube, 1}()
	qr = QuadratureRule{dim, RefCube}(2)
	cellvalues = CellScalarValues(qr, ip);

	dh = DofHandler(grid)
	push!(dh, :u, 1)
	close!(dh);

	K = create_sparsity_pattern(dh);
	M = create_sparsity_pattern(dh);

	f = zeros(ndofs(dh));

	ch = ConstraintHandler(dh);

	dOmega1 = union(getfaceset.((grid, ), ["left", "right","top", "bottom"])...)
	dbc = Dirichlet(:u, dOmega1, (x, t) -> 0)
	add!(ch, dbc);

	close!(ch)
	update!(ch, 0.0);


	K, f = doassemble_K!(K, f, cellvalues, dh)
	M = doassemble_M!(M, cellvalues, dh)
	A = (dt .* K) + M;

	rhsdata = get_rhs_data(ch, A);


	apply!(A, ch);

pvd = paraview_collection("transient-heat.pvd");

	print("Start Solving Heat Equation: Number of steps " * string(tsize-1) * "\n Done with step ")
	for t in 1:(tsize-1)
		#First of all, we need to update the Dirichlet boundary condition values.
		update!(ch, t)

		#Secondly, we compute the right-hand-side of the problem.
		b = dt .* f .+ M * un
		#Then, we can apply the boundary conditions of the current time step.
		apply_rhs!(rhsdata, b, ch)

		#Finally, we can solve the time step and save the solution afterwards.
		u = A \ b;
		
		#At the end of the time loop, we set the previous solution to the current one and go to the next time step.
		un .= u
    vtk_grid("test-heat-$t", dh) do vtk
        vtk_point_data(vtk, dh, u)
        vtk_save(vtk)
        pvd[t] = vtk
    end
		
		# move data into the tensor
		unmatrix= twistMatrix(reshape(un,(xsize,ysize)))
#		display(unmatrix)
		copyto!(output, t*xsize*ysize+ 1,  reshape(unmatrix,xsize*ysize), 1)
		print(string(t)*"\t")
	end
	print("\nFinish Solving Heat Equation\n")

vtk_save(pvd);
	return reshape(output,(xsize,ysize,tsize))
end