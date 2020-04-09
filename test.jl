struct Fragment
    elems::Array{String,1}
    coords::Array{Float64,2}
end

struct ORCAtheory
    inpline::String
    blocks::String
    charge::Int64
    mult::Int64
end

function Engrad(fragment,theory)
    println("Running Engrad")
    println("fragment is: $fragment")
    println("theory is: $theory")
    energy=0.0
    gradient=[0.0, 1.0, 0.0]
    return energy, gradient
end

function Optimizer(fragment,theory)
    println("Inside Optimizer")
    println("fragment is: $fragment")
    println("theory is: $theory")
    #Take initial step
    E, G = Engrad(fragment,theory)
    println("E: $E")
    println("G : $G")
    #Start loop
        #Take step

        #Call Engrad again

end


temp_el=["O","H","H"]
temp_coords=[[0.0, 0.0, 0.0] [0.0, 1.0, 1.0] [-1.0, -1.0, 0.0]]
charge_x=0
mult_x=1
orca_simpleinput="! BP86 def2-SVP"
orca_blocks="
%scf
maxiter 500
end
"
println(typeof(mult_x))

ORCAcalc=ORCAtheory(orca_simpleinput, orca_blocks, charge_x, mult_x)

molfrag=Fragment(temp_el,temp_coords)

println(molfrag)
println(molfrag.elems)
println(ORCAcalc.inpline)

#Engrad(molfrag,ORCAcalc)

Optimizer(molfrag,ORCAcalc)
