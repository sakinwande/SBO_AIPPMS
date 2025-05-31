function generate_s(pomdp::setFuncPOMDP, s::setFuncState, a::Symbol, rng::RNG)where {RNG <: AbstractRNG}
    #The only possible action is to sample a new point in the triangulation
    #The updated state immediate goes to that simplex

    #Convert the simplex to a point
    a_pos = POMDPs.actionindex(pomdp, a)
    pos = convert_pos_idx_2_pos_coord(pomdp, a_pos)

    push!(s.sampled_points, pos[1])

    #NOTE: the lower bound values have not been updated yet
    return setFuncState(a_pos, s.lb_func, s.ub_func, s.sampled_points)
end