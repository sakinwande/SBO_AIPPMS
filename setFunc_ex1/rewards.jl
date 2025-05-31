function belief_reward(pomdp::setFuncPOMDP, b::setFuncBelief, a::Symbol, bp::setFuncBelief)
    if b.lb_belief.X == []
        μ.init, ν.init, S_init = query_no_data(b.lb_belief)
    else
        μ.init, ν.init, S_init = query(b.lb_belief)
    end

    if b.ub_belief.X == []
        μ.init, ν.init, S_init = query_no_data(b.ub_belief)
    else
        μ.init, ν.init, S_init = query(b.ub_belief)
    end

    if bp.lb_belief.X == []
        μ.post, ν.post, S_post = query_no_data(bp.lb_belief)
    else
        μ.post, ν.post, S_post = query(bp.lb_belief)
    end

    if bp.ub_belief.X == []
        μ.post, ν.post, S_post = query_no_data(bp.ub_belief)
    else
        μ.post, ν.post, S_post = query(bp.ub_belief)
    end

    mu = μ_init[b.pos]
    σ = sqrt(ν_init[b.pos])
    z_score = 1.4

    if any([mu - z_score * σ <= mu+z_score*σ for d in b.sampled_points])
        r += pomdp.repeat_sample_penalty
    elseif round(μ_init[b.pos], digits=1) in b.sampled_points
        r += pomdp.repeat_sample_penalty
    elseif round(μ_init[b.pos], digits=1)==0.0 && 0.0 in b.sampled_points
        r += pomdp.repeat_sample_penalty
    else
        r += pomdp.new_sample_reward
    end

    variance_reduction = (sum(ν_init) - sum(ν_post))
    r += variance_reduction

    return r
end