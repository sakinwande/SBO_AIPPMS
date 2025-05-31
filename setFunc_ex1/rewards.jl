function belief_reward(pomdp::setFuncPOMDP, b::setFuncBelief, a::Symbol, bp::setFuncBelief)
    # query current lower bound GP
    if b.lb_belief.X == []
        μ_init, ν_init, S_init = query_no_data(b.lb_belief)
    else
        μ_init, ν_init, S_init = query(b.lb_belief)
    end

    # query current upper bound GP (results are currently unused but kept for completeness)
    if b.ub_belief.X == []
        μ_init_ub, ν_init_ub, S_init_ub = query_no_data(b.ub_belief)
    else
        μ_init_ub, ν_init_ub, S_init_ub = query(b.ub_belief)
    end

    # query posterior lower bound GP
    if bp.lb_belief.X == []
        μ_post, ν_post, S_post = query_no_data(bp.lb_belief)
    else
        μ_post, ν_post, S_post = query(bp.lb_belief)
    end

    # query posterior upper bound GP (unused)
    if bp.ub_belief.X == []
        μ_post_ub, ν_post_ub, S_post_ub = query_no_data(bp.ub_belief)
    else
        μ_post_ub, ν_post_ub, S_post_ub = query(bp.ub_belief)
    end

    mu = μ_init[b.pos]
    σ = sqrt(ν_init[b.pos])
    r = 0.0
    z_score = 1.4

    if any(mu - z_score*σ <= d <= mu + z_score*σ for d in b.sampled_points)
        r += pomdp.repeat_sample_penalty
    elseif round(μ_init[b.pos], digits=1) in b.sampled_points
        r += pomdp.repeat_sample_penalty
    elseif round(μ_init[b.pos], digits=1) == 0.0 && 0.0 in b.sampled_points
        r += pomdp.repeat_sample_penalty
    else
        r += pomdp.new_sample_reward
    end

    variance_reduction = sum(ν_init) - sum(ν_post)
    r += variance_reduction

    return r
end