struct Trajectory 
    S::Matrix{<:Real}
    L::Matrix{<:Real}
    T::Vector{<:Real}
end

struct TrajectoryMetrics
    Mean::Vector{<:Real}
    Var::Vector{<:Real}
    Std::Vector{<:Real}
    CV::Vector{<:Real}
    Skew::Vector{<:Real}
end

"""
    ssa_step(system::NamedTuple,trans_mat::AbstractMatrix{<:Real},changing_mat::AbstractMatrix{<:Real})

Performs a single step of **n** state Gillespie SSA.
"""
function ssa_step(system::NamedTuple,
    trans_mat::AbstractMatrix{<:Real},
    changing_mat::AbstractMatrix{<:Real})

    r,c = size(trans_mat)
    @assert r == c "Transition matrix must be square"
    @assert size(changing_mat,2) == c "Size changing matrix must have same number of columns as transition matrix"
    @assert size(changing_mat,1) == 2 "Size changing matrix must have only two rows "
    if system.S == 0
        return system
    elseif 1 <= system.S <= r 
        rates = vcat(trans_mat[system.S, :], changing_mat[:, system.S])
        @assert all(rates .>= 0) "All rates must be non-negative"
        @assert sum(rates) > 0 "Total propensity must be positive"
        probs = cumsum(rates) ./ sum(rates)
        dt = - log(rand()) / sum(rates)
        rnd = rand(); state = system.S; len = system.L
        for i in 1:length(probs)
            if rnd <= probs[i]
                if (1 <= i <= r) & (i == system.S)
                    state = 0
                elseif (1 <= i <= r) & (i != system.S)
                    state = i
                elseif i == (r+1)
                    len += 1
                elseif i == (r+2)
                    len -= 1                
                end
                break
            end
        end
        return (S = state, time = system.time + dt, L = len)
    else
        throw("State is not recognized")      
    end
end

function ssa_step(system::NamedTuple,
    trans_mat::AbstractMatrix{<:Real})

    r,c = size(trans_mat)
    return ssa_step(system, trans_mat, zeros(2,c))
end

function ssa_N(trans_mat::AbstractMatrix{<:Real},
    changing_mat::AbstractMatrix{<:Real}; 
    T_max::Real = 1e2, L_init::Integer=0, t_init::Real = 0.0, 
    N::Integer = 1, S_init::Integer = 1, T_step::Real=0.1)::Trajectory

    T = collect(t_init:T_step:T_max)
    L = zeros(N,length(T)) .+ L_init;
    S = S_init .* ones(N,length(T));
    for i in 1:N
        syst = (S=S_init, L=L_init, time = t_init)
        while syst.time < T_max
            syst = ssa_step(syst, trans_mat, changing_mat)
            L[i, T .>= syst.time] .= syst.L
            S[i, T .>= syst.time] .= syst.S
            if syst.S == 0
                break
            end
        end
    end
    return Trajectory(S,L,T)
end
function ssa_N(trans_mat::AbstractMatrix{<:Real}; 
    T_max::Real = 1e2, L_init::Integer=0, t_init::Real = 0.0, 
    N::Integer = 1, S_init::Integer = 1, T_step::Real=0.1)::Trajectory

    T = collect(t_init:T_step:T_max)
    L = zeros(N,length(T)) .+ L_init;
    S = S_init .* ones(N,length(T));
    for i in 1:N
        syst = (S=S_init, L=L_init, time = t_init)
        while syst.time < T_max
            syst = ssa_step(syst, trans_mat)
            L[i, T .>= syst.time] .= syst.L
            S[i, T .>= syst.time] .= syst.S
            if syst.S == 0
                break
            end
        end
    end
    return Trajectory(S,L,T)
end

function traj_metric(traj::Trajectory)::TrajectoryMetrics    
    L_mean = mean(traj.L, dims=1) |> vec;
    L_var = var(traj.L, dims=1) |> vec;
    L_std = std(traj.L, dims=1) |> vec;
    L_CV = L_std ./ L_mean;
    L_skw = [skewness(traj.L[:,i]) for i in 1:length(traj.T)]
    return TrajectoryMetrics(L_mean, L_var, L_std, L_CV, L_skw)
end

function traj_prob(traj::Trajectory, state::Integer)
    N = size(traj.S, 1)
    return (count((i)->(i==state), traj.S, dims = 1) ./ N)|> vec
end