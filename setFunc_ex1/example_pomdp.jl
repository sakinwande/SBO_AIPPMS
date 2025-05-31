using POMDPModels, POMDPs, POMDPPolicies
using StaticArrays, Parameters, Random, POMDPModelTools, Distributions
using Images, LinearAlgebra, Printf
using Plots
using StatsBase
using BasicPOMCP


#Take inspiration from the rover problem to define the evaluation problem as a belief MDP

const setFuncPos = SVector{1,Float64}
#The state is point, a set of sampled points, and function values at the sampled points 
struct setFuncState
    pos::Int
    lb_func
    ub_func
    sampled_points::Vector{Float64}
end

#For the particle filter
struct particleSet
    lb_particles::Matrix{Float64}
    lb_weights::Matrix{Float64}
    ub_particles::Matrix{Float64}
    ub_weights::Matrix{Float64}
end

#Represent the belief as Gaussian Processes 
struct setFuncBelief
    pos::Int
    lb_belief::GaussianProcess
    ub_belief::GaussianProcess
    sampled_points::Vector{Float64}
end

@with_kw mutable struct setFuncPOMDP <: POMDP{setFuncState, Symbol, Float64}    
    oracle
    true_lbs
    true_ubs
    domain
    num_simps::Int        
    lb_prior::GaussianProcess
    ub_prior::GaussianProcess
    rng::AbstractRNG
    init_pos::Tuple{Int,}              = (1,) #this should be a random sample?
    map_size::Tuple{Int,Int}              = (num_simps,1) # size of the map

    #TODO: Revisit numbers below here
    σ_drill::Float64                       = 1e-9
    step_size::Int64                       = 1 # scales the step the agent takes at each iteration
    new_sample_reward::Float64             = 1 # reward for drilling a unique sample
    repeat_sample_penalty::Float64         = -1 # penalty for drilling a repeat sample
    aind::Dict{Symbol,Int}
end

function POMDPs.gen(pomdp::setFuncPOMDP, s::setFuncState, a::Symbol, rng::RNG)where {RNG <: AbstractRNG}
    sp = generate_s(pomdp, s, a, rng)
    o = generate_o(pomdp, s, a, sp, rng)
    r = reward(pomdp, s, a, sp)

    return (sp=sp, o=o, r=r)
end

#TODO: Consider adding budget check for terminal states
function POMDPs.isterminal(pomdp::setFuncPOMDP, s::setFuncState)
    return false
end

function POMDPs.isterminal(pomdp::setFuncPOMDP, b::setFuncBelief)
    return false
end
include("states.jl")
include("actions.jl")
include("observations.jl")
include("beliefs.jl")
include("transitions.jl")
include("rewards.jl")
