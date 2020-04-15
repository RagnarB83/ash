#Some Julia functions
__precompile__()

module Juliafunctions

function hellofromjulia()
println("Hello from Julia")
end

function juliatest(list)
println("Inside juliatest")
println("list is : $list")
var=5.4
return var
end




end