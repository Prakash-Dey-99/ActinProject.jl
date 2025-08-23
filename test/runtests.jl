using ActinProject
using Test
using DataFrames, LsqFit

@testset "ActinProject.jl" begin
    
    data = DataFrame(X=1:10, Y=5 .* (1:10) .+ rand(10));
    func(x,p) = @. p[1]*x + p[2];
    p_init = [0.0,0.0]
    fit_res = fit_and_calculate(data, :X, :Y, func, p_init);

    @test typeof(fit_res) == NamedTuple{(:params, :r_param, :r_err, :prediction, :ribb, :model),
                                       Tuple{Vector{Float64},
                                             Vector{Float64},
                                             Vector{Float64},
                                             Vector{Float64},
                                             Tuple{Vector{Float64}, Vector{Float64}},
                                             LsqFit.LsqFitResult{Vector{Float64}, Vector{Float64}, Matrix{Float64}, Vector{Float64}, 
                                             Vector{LsqFit.LMState{LsqFit.LevenbergMarquardt}}}
                                             }}
end
