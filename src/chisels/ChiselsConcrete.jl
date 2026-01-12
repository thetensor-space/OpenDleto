function TuckerChisel(axises::Vector{<:ITensors.Index}, ch_axis::ITensors.Index)::Chisel
    m = LinearAlgebra.Diagonal( [1.0 for _=1:length(axises)] )
    @assert size(m,1)  == ITensors.dim(ch_axis) "Incompatable axis"
    return Chisel(m,axises,ch_axis)
end;

TuckerChisel(axises::Vector{<:ITensors.Index})::Chisel = TuckerChisel(axises, ITensors.Index(length(axises), "chisel,Tucker"));

TuckerChisel(d::Dict{<:ITensors.Index, Bool})::Chisel = 
    extend_chisel( 
        TuckerChisel([ i for i in keys(d) if d[i]]),
        [ i for i in keys(d)]
    );

function UniversalChisel(axises::Vector{<:ITensors.Index}, ch_axis::ITensors.Index)::Chisel
    m = LinearAlgebra.transpose( [1.0 for _=1:length(axises)] )
    @assert size(m,1)  == ITensors.dim(ch_axis) "Incompatable axis"
    return Chisel(m,axises,ch_axis)
end;

UniversalChisel(axises::Vector{<:ITensors.Index})::Chisel = UniversalChisel(axises, ITensors.Index(1, "chisel,Universal"));

UniversalChisel(d::Dict{<:ITensors.Index, Bool})::Chisel = 
    extend_chisel( 
        UniversalChisel([ i for i in keys(d) if d[i]]),
        [ i for i in keys(d)]
    );

function CentroidChisel(axises::Vector{<:ITensors.Index}, ch_axis::ITensors.Index)::Chisel
    e = length(axises)
    m = zeros( e*(e-1) ÷ 2, e )
    row = 1    
    for a =2:e
        for b = 1:(a-1)
            m[row, a] = 1.0
            m[row, b] = -1.0
            row += 1
        end
    end    
    @assert size(m,1)  == ITensors.dim(ch_axis) "Incompatable axis"
    return Chisel(m,axises,ch_axis)
end;

CentroidChisel(axises::Vector{<:ITensors.Index})::Chisel = 
    CentroidChisel(axises, ITensors.Index(length(axises)*(length(axises)-1)÷ 2, "chisel,Centroid"));

CentroidChisel(d::Dict{<:ITensors.Index, Bool})::Chisel = 
    extend_chisel( 
        CentroidChisel([ i for i in keys(d) if d[i]]),
        [ i for i in keys(d)]
    );


AdjointChisel(left::ITensors.Index, right::ITensors.Index, ch_axis::ITensors.Index)::Chisel = CentroidChisel([left,right],ch_axis);
AdjointChisel(left::ITensors.Index, right::ITensors.Index)::Chisel = CentroidChisel([left,right], ITensors.Index(1,"chisel,Adjoint"));
