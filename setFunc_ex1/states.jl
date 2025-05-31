# function POMDPs.initialstate(pomdp::setFuncPOMDP, rng::AbstractRNG)
#     return setFuncState(0, pomdp.true_lbs, pomdp.true_ubs, Set{Float64}(Float64[]))
# end

function convert_pos_idx_2_pos_coord(pomdp::setFuncPOMDP, pos::Int)
    Interval = pomdp.domain[2] - pomdp.domain[1]
    step_size = Interval / pomdp.num_simps

    int_start = pomdp.domain[1] + step_size * (pos - 1)
    int_end = pomdp.domain[1] + step_size * pos

    #uniformly sample from the interval
    return setFuncPos(rand(pomdp.rng, Distributions.Uniform(int_start, int_end)))
end

function convert_pos_coord_2_pos_idx(pomdp::setFuncPOMDP, pos::setFuncPos)
    #Return the index of the interval that contains the position
    Interval = pomdp.domain[2] - pomdp.domain[1]
    step_size = Interval / pomdp.num_simps
    idx = findfirst(x -> x[1] <= pos < x[2], [pomdp.domain[1] + step_size * (i - 1), pomdp.domain[1] + step_size * i] for i in 1:pomdp.num_simps)

    return idx
end

function POMDPs.initialstate(pomdp::setFuncPOMDP)
    curr = rand(pomdp.rng, 1:pomdp.num_simps)
    return setFuncState(curr, pomdp.true_lbs, pomdp.true_ubs, Float64[])
end

