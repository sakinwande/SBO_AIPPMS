struct setFuncBeliefUpdater{P<:POMDPs.POMDP} <: Updater
    pomdp::P
end

function Base.rand(rng::AbstractRNG, pomdp::setFuncPOMDP, b::setFuncBelief)
    lb_states = rand(rng, b.lb_belief, b.lb_belief.mXq, b.lb_belief.KXqXq)
    ub_states = rand(rng, b.ub_belief, b.ub_belief.mXq, b.ub_belief.KXqXq)
    lb_states = reshape(lb_states, pomdp.map_size)
    ub_states = reshape(ub_states, pomdp.map_size)

    return setFuncState(b.pos, lb_states, ub_states, b.sampled_points)
end

#TODO: Update observation to be a set of samples from a simplex
"""
    POMDPs.update(updater::setFuncBeliefUpdater, b::setFuncBelief, a::Symbol, o)

Update the belief given a symbolic action and an observation.  The solver
produces actions as `Symbol`s, so we convert to an action index and then call
`update_belief` which performs the Gaussian process posterior update.
"""
function POMDPs.update(updater::setFuncBeliefUpdater, b::setFuncBelief, a::Symbol, o)
    a_idx = POMDPs.actionindex(updater.pomdp, a)
    return update_belief(updater.pomdp, b, a_idx, o, updater.pomdp.rng)
end

#TODO:
function update_belief(pomdp::P, b::setFuncBelief, a::Int, o, rng::RNG) where {P <: POMDPs.POMDP, RNG <: AbstractRNG}
    # `o` is a vector of the form [pos, observation]
    pos_coord = o[1]
    obs_val = o[2]

    σ²_n = pomdp.σ_drill

    lb_posterior = posterior(b.lb_belief, [pos_coord[1]], [obs_val], [σ²_n])
    ub_posterior = posterior(b.ub_belief, [pos_coord[1]], [obs_val], [σ²_n])

    new_eval_samples = copy(b.sampled_points)
    push!(new_eval_samples, obs_val)

    sample_pos = convert_pos_coord_2_pos_idx(pomdp, pos_coord)
    return setFuncBelief(sample_pos, lb_posterior, ub_posterior, new_eval_samples)
end

function BasicPOMCP.extract_belief(::setFuncBeliefUpdater, node::BeliefNode)
    return node
end

function POMDPs.initialize_belief(updater::setFuncBeliefUpdater,d)
    return initial_belief_state(updater.pomdp, updater.pomdp.rng)
end

function POMDPs.initialize_belief(updater::setFuncBeliefUpdater, d,rng::AbstractRNG)
    return initial_belief_state(updater.pomdp, rng)
end

function initial_belief_state(pomdp::setFuncPOMDP, rng::RNG) where {RNG <: AbstractRNG}
    pos = LinearIndices(pomdp.map_size)[rand(rng, 1:prod(pomdp.map_size))]
    lb_belief = pomdp.lb_prior
    ub_belief = pomdp.ub_prior
    eval_samples = Float64[pos]

    return setFuncBelief(pos, lb_belief, ub_belief, eval_samples)
end

