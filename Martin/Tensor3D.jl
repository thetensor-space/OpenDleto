########################################################################
##  CC-BY 2021 Peter A. Brooksbank, Martin Kassabov, James B. Wilson
##
## MIT License
########################################################################


function mycolor( val, norm ,factor)
    d = 0.5
    n = sqrt(abs(val)/norm) * factor
    #n = round(d*n, digits=4)
    #n = 0.5
    return "255 0 0", [d*n,d*n,d*n], [n,n,n]
end


function rstring(x)
    return string(round(x, digits=4))
end

function printBoxCorner(  x,y,z, val, norm,factor )
    xwidth = 1; ywidth = 1; zwidth = 1;

    s = ""
    col, shift, widths = mycolor(val, norm,factor)
    # 0 0 0
    s *= rstring(xwidth*x-shift[1]) * " " * rstring(ywidth*y-shift[2]) * " " * rstring(zwidth*z-shift[3]) * " " * col * "\n"
    # 0 0 1
    s *= rstring(xwidth*x-shift[1]) * " " * rstring(ywidth*y-shift[2]) * " " * rstring(zwidth*(z+widths[3])-shift[3]) * " " * col * "\n"
    # 0 1 1
    s *= rstring(xwidth*x-shift[1]) * " " * rstring(ywidth*(y+widths[2])-shift[2]) * " " * rstring(zwidth*(z+widths[3])-shift[3]) * " " * col * "\n"
    # 0 1 0
    s *= rstring(xwidth*x-shift[1]) * " " * rstring(ywidth*(y+widths[2])-shift[2]) * " " * rstring(zwidth*z-shift[3]) * " " * col * "\n"
    # 1 0 0
    s *= rstring(xwidth*(x+widths[1])-shift[1]) * " " * rstring(ywidth*y-shift[2]) * " " * rstring(zwidth*z-shift[3]) * " " * col * "\n"
    # 1 0 1
    s *= rstring(xwidth*(x+widths[1])-shift[1]) * " " * rstring(ywidth*y-shift[2]) * " " * rstring(zwidth*(z+widths[3])-shift[3]) * " " * col * "\n"
    # 1 1 1
    s *= rstring(xwidth*(x+widths[1])-shift[1]) * " " * rstring(ywidth*(y+widths[2])-shift[2]) * " " * rstring(zwidth*(z+widths[3])-shift[3]) * " " * col * "\n"
    # 1 1 0
    s *= rstring(xwidth*(x+widths[1])-shift[1]) * " " * rstring(ywidth*(y+widths[2])-shift[2]) * " " * rstring(zwidth*z-shift[3]) * " " * col * "\n"
    return s
end

printBoxFace = function( num )
    s = ""
    # 4 verts: 0 1 2 3                 
    s *= string(4) * " " * string(0+8*num) * " " * string(1+8*num) * " " * string(2+8*num) * " " * string(3+8*num) * "\n"
    # 4 verts: 7 6 5 4
    s *= string(4) * " " * string(7+8*num) * " " * string(6+8*num) * " " * string(5+8*num) * " " * string(4+8*num) * "\n"
    # 4 verts: 0 4 5 1
    s *= string(4) * " " * string(0+8*num) * " " * string(4+8*num) * " " * string(5+8*num) * " " * string(1+8*num) * "\n"
    # 4 verts: 1 5 6 2
    s *= string(4) * " " * string(1+8*num) * " " * string(5+8*num) * " " * string(6+8*num) * " " * string(2+8*num) * "\n"
    # 4 verts: 2 6 7 3
    s *= string(4) * " " * string(2+8*num) * " " * string(6+8*num) * " " * string(7+8*num) * " " * string(3+8*num) * "\n"
    # 4 verts: 3 7 4 0
    s *= string(4) * " " * string(3+8*num) * " " * string(7+8*num) * " " * string(4+8*num) * " " * string(0+8*num) * "\n"

    return s
end


function print3D(t,ratio,factor)
#    norm = round(max( abs(maximum(t)), abs(minimum(t))), digits=4)  ## could be more intellegent but I wont be
# this is safer
	norm =0
    for i = axes(t,1)
        for j = axes(t,2)
            for k = axes(t,3)
                val = abs(t[i,j,k])
                if (val > norm) 
					norm = val
                end
            end 
        end 
    end 


    verts = ""
    faces = ""
	raw = "x y z meta size\n"
    count = 0
    for i = axes(t,1)
		vertst = ""
		facest = ""
		rawt =""
        for j = axes(t,2)
			vertstt = ""
			facestt = ""
			rawtt = "" 
            for k = axes(t,3)
                val = abs(t[i,j,k])
                if (val > norm/ ratio) 
                    vertstt *= printBoxCorner(i,j,k, val, norm,factor)
                    facestt *= printBoxFace(count)
					rawtt *= string(i) * " " * string(j) * " " * string(k) * " 0 " * string(trunc(Int,ratio *val/norm )) * "\n"
                    count += 1
                end
            end
			vertst *= vertst
			facest *= facestt
			rawt *= rawtt
        end 
		verts *= verts
		faces *= facest
		raw *= rawt
    end 

    ply = "ply\n"
    ply *= "comment made by Magma  { Tensor rendered by Julia System }\n"
    ply *= "format ascii 1.0\n"
    ply *= "element vertex " * string(8*count) * "\n"
    ply *= "property float x\n"
    ply *= "property float y\n"
    ply *= "property float z\n"
    ply *= "property uchar red                   { start of vertex color }\n"
    ply *= "property uchar green\n"
    ply *= "property uchar blue\n"
    ply *= "element face " * string(6*count) * "\n"
    ply *= "property list uint8 int32 vertex_index\n"
    ply *= "end_header\n"

    ply *= verts
    ply *= faces
    return ply, raw
end 

#add a raw output
# factors makes cubes in the output larger
function save3D(t, nameply,nameraw,ratio,factor)
	ply, raw = print3D(t,ratio,factor)
    fileply = open(nameply, "w")
    write(fileply, ply)
    close(fileply)
    fileraw = open(nameraw, "w")
    write(fileraw, raw)
    close(fileraw)
end



#Function the surface in the graph
#s(x,y) = abs(((1-y)^2.5 - x^2.5 + 0.5))^0.4* sign((1-y)^2.5 - x^2.5 + 0.5)

# produce a ply for a function
#
function graph3D(fun,xsize,ysize,zsize,step,name)
	m = zeros(Int64,(step+1,step+1))
	spoints = ""
	sfaces = ""
	pcount = 0
	fcount = 0
	for i = 1:(step+1)
		s = ""
		for j = 1:(step+1)
			z = fun( (i-1)/step, (j-1)/step )
			if ((z >= 0) && (z <= 1))
				pcount +=1
				m[i,j]= pcount
				#generate point
				s *= rstring(1+(i-1)*(xsize-1)/step) * " " * rstring(1+(j-1)*(ysize-1)/step) * " " * rstring(1+z*(zsize-1)) * " 255 0 0 \n"
			end
		end
		spoints *= s
	end
	for i = 1:step
		s = "" 
		for j = 1:step
			if (m[i,j] > 0)  && (m[i+1,j] > 0 ) && (m[i+1,j+1] > 0 ) && (m[i,j+1] > 0 )
				fcount +=1
				#generate face
				s *= "4 " * string(m[i,j]-1) * " " * string(m[i+1,j]-1) * " " * string(m[i+1,j+1]-1) * " " * string(m[i,j+1]-1) * " \n"
			else
				if (m[i,j] > 0)  && (m[i+1,j] > 0 ) && (m[i+1,j+1] > 0 )
					fcount +=1
					#generate face
					s *= "3 " * string(m[i,j]-1) * " " * string(m[i+1,j]-1) * " " * string(m[i+1,j+1]-1) * " \n"
				end
				if (m[i,j] > 0)  && (m[i+1,j] > 0 ) && (m[i,j+1] > 0 )
					fcount +=1
					#generate face
					s *= "3 " * string(m[i,j]-1) * " " * string(m[i+1,j]-1) * " " * string(m[i,j+1]-1) * " \n"
				end
				if (m[i,j] > 0)  && (m[i+1,j+1] > 0 ) && (m[i,j+1] > 0 )
					fcount +=1
					#generate face
					s *= "3 " * string(m[i,j]-1) * " " * string(m[i+1,j+1]-1) * " " * string(m[i,j+1]-1) * " \n"
				end
				if (m[i+1,j] > 0)  && (m[i+1,j+1] > 0 ) && (m[i,j+1] > 0 )
					fcount +=1
					#generate face
					s *= "3 " * string(m[i+1,j]-1) * " " * string(m[i+1,j+1]-1) * " " * string(m[i,j+1]-1) * " \n"
				end
			end
		end
		sfaces *= s
	end
	#combine
    ply = "ply\n"
    ply *= "comment made by Magma  { Tensor rendered by Julia System }\n"
    ply *= "format ascii 1.0\n"
    ply *= "element vertex " * string(pcount) * "\n"
    ply *= "property float x\n"
    ply *= "property float y\n"
    ply *= "property float z\n"
    ply *= "property uchar red                   { start of vertex color }\n"
    ply *= "property uchar green\n"
    ply *= "property uchar blue\n"
    ply *= "element face " * string(fcount) * "\n"
    ply *= "property list uint8 int32 vertex_index\n"
    ply *= "end_header\n"

    ply *= spoints
    ply *= sfaces

    file = open(name, "w")
    write(file, ply)
    close(file)
#    return ply
end

