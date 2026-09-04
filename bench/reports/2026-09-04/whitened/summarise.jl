#
# Turn bench/reports/2026-09-04/whitened/whitened.csv into the markdown table
# in README.md next to it: one row per (case, solver) with the unwhitened and
# whitened numbers side by side and the ratios.
#
#   bench/jl bench/reports/2026-09-04/whitened/summarise.jl
#
using Printf

const DIR = @__DIR__
const CSV = joinpath(DIR, "whitened.csv")

function rows()
    lines = filter(l -> !startswith(l, "#") && !isempty(strip(l)), readlines(CSV))
    hdr = split(lines[1], ',')
    out = Dict{String,String}[]
    for l in lines[2:end]
        # only the dims/r fields are quoted, and neither contains a comma we
        # need to keep, so strip quotes and split on commas outside them
        f = String[]
        buf = IOBuffer(); inq = false
        for ch in l
            if ch == '"'
                inq = !inq
            elseif ch == ',' && !inq
                push!(f, String(take!(buf)))
            else
                print(buf, ch)
            end
        end
        push!(f, String(take!(buf)))
        length(f) == length(hdr) || continue
        push!(out, Dict(zip(hdr, f)))
    end
    return out
end

num(s) = isempty(s) ? NaN : something(tryparse(Float64, s), NaN)
ratio(a, b) = (isnan(a) || isnan(b) || b == 0) ? "--" : @sprintf("%.2fx", a / b)

function main()
    rs = rows()
    # last run wins for a repeated (case, solver, whiten)
    byk = Dict{Tuple{String,String,String},Dict{String,String}}()
    order = Tuple{String,String}[]
    for r in rs
        k = (r["case"], r["solver"], r["whiten"])
        haskey(byk, k) || push!(order, (r["case"], r["solver"]))
        byk[k] = r
    end
    seen = Set{Tuple{String,String}}()
    io = IOBuffer()
    println(io, "| case | dims | r | solver | applies (plain -> whitened) | iter gain | ",
                "seconds | time gain | peak RSS GB | nullity (oracle) | Z-law residual | verdict |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (case, solver) in order
        (case, solver) in seen && continue
        push!(seen, (case, solver))
        p = get(byk, (case, solver, "0"), nothing)
        w = get(byk, (case, solver, "1"), nothing)
        (p === nothing || w === nothing) && continue
        verdict(r) = r["status"] != "ok" ? "FAILED (" * first(split(r["status"], ':')) * ")" :
                     num(r["nullity"]) == num(r["oracle_nullity"]) ? "ok" :
                     "WRONG nullity"
        @printf(io, "| %s | %s | %s | %s | %d -> %d | %s | %.1f -> %.1f | %s | %.2f / %.2f | %d / %d (%s) | %.1e / %.1e | %s / %s |\n",
                case, w["dims"], w["r"], solver,
                Int(num(p["applies"])), Int(num(w["applies"])),
                ratio(num(p["applies"]), num(w["applies"])),
                num(p["seconds"]), num(w["seconds"]),
                ratio(num(p["seconds"]), num(w["seconds"])),
                num(p["maxrss_GB"]), num(w["maxrss_GB"]),
                Int(num(p["nullity"])), Int(num(w["nullity"])), w["oracle_nullity"],
                num(p["residual"]), num(w["residual"]),
                verdict(p), verdict(w))
    end
    print(String(take!(io)))
    return nothing
end

main()
