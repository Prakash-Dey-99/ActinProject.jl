using ActinProject
using Test
using DataFrames, LsqFit

@testset "ActinProject.jl" begin
    # Test for fitting
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
    @test length(fit_res.params) == 2
    @test length(fit_res.r_param) == 2
    @test length(fit_res.r_err) == 2
    @test length(fit_res.prediction) == 10
    @test length(fit_res.ribb[1]) == 10
    @test length(fit_res.ribb[2]) == 10 

    # test for SSA
    tran = [0.5 1.0; 0.2 0.1]; grow = [0.6 6.0; 0.0 0.0];
    traj = ssa_N(tran, grow, T_max=10.0, L_init=0, t_init=0.0, N=5, S_init=1, T_step=0.5);
    @test typeof(traj) == Trajectory
    @test size(traj.S) == (5,21)
    @test size(traj.L) == (5,21)
    @test length(traj.T) == 21

    metrics = traj_metric(traj);
    @test typeof(metrics) == TrajectoryMetrics
    @test length(metrics.Mean) == 21
    @test length(metrics.Var) == 21
    @test length(metrics.Std) == 21
    @test length(metrics.CV) == 21
    @test length(metrics.Skew) == 21
end
