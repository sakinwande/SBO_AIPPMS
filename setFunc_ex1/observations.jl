function generate_o(pomdp::setFuncPOMDP, s::setFuncState, action::Symbol, sp::setFuncState, rng::AbstractRNG)
    a_pos = POMDPs.actionindex(pomdp, action)
    pos = convert_pos_idx_2_pos_coord(pomdp, a_pos)
    o = pomdp.oracle(pos)
    return [pos,o]
end