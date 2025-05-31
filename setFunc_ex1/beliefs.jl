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
function POMDPs.update(updater::setFuncBeliefUpdater, b::setFuncBelief, a::Int, o::Float64)
    ub = update_belief(updater.pomdp, b, a, o, updater.pomdp.rng)
    return ub
end

#TODO:
function update_belief(pomdp::P, b::setFuncBelief, a::Int, o::Float64, rng::RNG) where {P <: POMDPs.POMDP, RNG <: AbstractRNG}
    # NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
    # for normal dist whereas our GP setup uses σ²_n

    # if we sample we fully collapse the belief
    σ²_n = pomdp.σ_drill #^2 dont square this causes singular exception in GP update
    lb_posterior = posterior(b.lb_belief, [sample_pos[1]], [o[2]], [σ²_n])
    ub_posterior = posterior(b.ub_belief, [sample_pos[1]], [o[2]], [σ²_n])

    new_eval_samples = push!(b.sampled_points, o[2])
    sample_pos = convert_pos_coord_2_pos_idx(pomdp, sample_pos)

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

