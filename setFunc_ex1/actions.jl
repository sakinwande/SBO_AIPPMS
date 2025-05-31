function POMDPs.actions(pomdp::setFuncPOMDP, s::setFuncState)
    possible_actions = collect(1:pomdp.map_size[1])
    return Symbol.(possible_actions)
end

function POMDPs.action(p::RandomPolicy, s::setFuncState)
    possible_actions = POMDPs.actions(p.problem, s)
    return rand(p.problem.pomdp.rng, Symbol.(possible_actions)
    )
end

function POMDPs.actions(pomdp::setFuncPOMDP, b::setFuncBelief)
    possible_actions = collect(1:pomdp.map_size[1])
    return Symbol.(possible_actions)

end

function POMDPs.actions(pomdp::setFuncPOMDP, s::LeafNodeBelief)
    possible_actions = collect(1:pomdp.map_size[1])
    return Symbol.(possible_actions)

end

