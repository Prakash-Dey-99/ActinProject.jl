module ActinProject

using LsqFit, DataFrames, Statistics, StatsBase

export fit_and_calculate
export TrajectoryMetrics, Trajectory
export ssa_step, ssa_N, traj_metric, traj_prob

include("Fitting.jl")
include("SSA.jl")

end
