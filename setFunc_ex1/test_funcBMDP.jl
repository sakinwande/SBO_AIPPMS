include("GP.jl")
include("example_pomdp.jl")
include("belief_mdp.jl")
using Random
using BasicPOMCP
using POMDPs
using Statistics
using Distributions
using Plots
using KernelFunctions
using MCTS
using DelimitedFiles

#We don't have a terminal state
POMDPs.isterminal(bmdp::BeliefMDP, b::setFuncBelief) = false

domain = [-10,4]
function lb_func(x, ϵ = 5e-2)
    x = x[1]
    return sin(x) - ϵ*(x^2 - 2*x +2)
end

function ub_func(x, ϵ = 5e-2)
    x = x[1]
    return sin(x) + ϵ*(x^2 - 2*x +2)
end

function oracle_func(x, ϵ = 5e-2)
    oracleInt = [
        lb_func(x, ϵ),
        ub_func(x, ϵ)
    ]

    return rand(Distributions.Uniform(oracleInt[1], oracleInt[2]))
end

function run_setFunc_bmdp(rng::RNG, bmdp::BeliefMDP, policy, n_iters=20) where {RNG<:AbstractRNG}

    belief_state = initialstate(bmdp).val

    # belief_state = initial_belief_state(bmdp, rng)
	state_hist = [deepcopy(belief_state.pos)]
	lb_gp_hist = [deepcopy(belief_state.lb_belief)]
    ub_gp_hist = [deepcopy(belief_state.ub_belief)]
	action_hist = []
	reward_hist = []
	total_reward_hist = []
	total_planning_time = 0

    total_reward = 0.0

    i = 0
    while i<= n_iters
        a, t = @timed policy(belief_state)

		total_planning_time += t

		new_belief_state, sim_reward = POMDPs.gen(bmdp, belief_state, a, rng)

		# just use these to get the true reward NOT the simulated reward
		s = setFuncState(belief_state.pos, bmdp.pomdp.true_map, belief_state.cost_expended, belief_state.drill_samples)
		sp = setFuncState(new_belief_state.pos, bmdp.pomdp.true_map, new_belief_state.cost_expended, new_belief_state.drill_samples)
		true_reward = reward(bmdp.pomdp, s, a, sp)

        total_reward += true_reward
        belief_state = new_belief_state

		state_hist = vcat(state_hist, deepcopy(belief_state.pos))
		gp_hist = vcat(gp_hist, deepcopy(belief_state.location_belief))

		action_hist = vcat(action_hist, deepcopy(a))
		reward_hist = vcat(reward_hist, deepcopy(true_reward))
		total_reward_hist = vcat(total_reward_hist, deepcopy(total_reward))

        i += 1
    end

    return total_reward, state_hist, lb_gp_hist, ub_gp_hist, action_hist, reward_hist, total_reward_hist, total_planning_time, length(reward_hist)

end

function get_gp_bmdp_policy(bmdp, rng, max_depth=20, queries=100)
    planner = solve(MCTS.DPWSolver(depth=max_depth, n_iterations=queries, rng=rng, k_state=0.5, k_action=10000.0, alpha_state=0.5), bmdp)

    return b -> action(planner, b)
end

seed=1234
n_trials=5
n_simps=10
# function solver_test_setFuncBMDP(seed::Int64=1234, n_trials=5, n_simps=10)
    k = with_lengthscale(SqExponentialKernel(), 1.0)
    m(x) = 0.0

    X_query = [i for i=1:n_simps]
    query_size = size(X_query)
    X_query = reshape(X_query, query_size[1]*1)
    KXqXq = K(X_query, X_query, k)

    μ(X_query,m)

    LB_GP = GaussianProcess(m, μ(X_query, m), k, [], X_query, [], [], [], [], KXqXq);
    UB_GP = GaussianProcess(m, μ(X_query, m), k, [], X_query, [], [], [], [], KXqXq);

    lb_prior = LB_GP
    ub_prior = UB_GP

    gp_mcts_rewards = Vector{Float64}(undef, 0)
	rmse_hist_gp_mcts = []
	trace_hist_gp_mcts = []
	total_planning_time_gp_mcts = 0
	total_plans_gp_mcts = 0

    
    
    i = 1
    idx = 1

    # while idx <= n_trials
        @show i
        rng = MersenneTwister(seed) #TODO: add +I

        oracle = oracle_func
        aind = Dict(Symbol(i) => i for i in 1:n_simps)
        POMDPs.actionindex(pomdp::setFuncPOMDP, a::Symbol) = aind[a]

        pomdp = setFuncPOMDP(oracle, lb_func, ub_func, domain, n_simps, lb_prior, ub_prior, rng, (1,), (10, 1), 1e-9, 1, 1, -1, aind)

        bmdp = BeliefMDP(pomdp, setFuncBeliefUpdater(pomdp), belief_reward)

        gp_bmdp_isterminal(s) = POMDPs.isterminal(pomdp, s)
        depth = 5
        gp_bmdp_policy = get_gp_bmdp_policy(bmdp, rng, depth)

        #GP-MCTS-DPW
        gp_mcts_reward, state_hist, gp_hist, action_hist, reward_hist, total_reward_hist, planning_time, num_plans = run_setFunc_bmdp(rng, bmdp, gp_bmdp_policy)
        total_planning_time_gp_mcts += planning_time
        println("average planning time: ", planning_time/num_plans)
		total_plans_gp_mcts += num_plans
		rmse_hist_gp_mcts = vcat(rmse_hist_gp_mcts, [calculate_rmse_along_traj(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i)])
		trace_hist_gp_mcts = vcat(trace_hist_gp_mcts, [calculate_trace_Σ(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i)])

        @show gp_mcts_reward

        i = i + 1
        idx = idx + 1

        push!(gp_mcts_rewards, gp_mcts_reward)
    end


	println("GP-MCTS-DPW average planning time: ", total_planning_time_gp_mcts/total_plans_gp_mcts)
	@show mean(gp_mcts_rewards)
#end


boo = [1,2,3,4]

symbols = Symbol.(string.(boo))

println(symbols)

typeof(symbols[1])

typeof(:jk)