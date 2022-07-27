using Crux, Distributions
include("utils.jl")
include("../environments/pendulum_problem.jl")

## Parameters
# Experiment params
Neps=100_000
Neps_gt=1_000_000
Ntrials=5
dir="results/pendulum_discrete_vb_fixed_logits/"

# Problem setup params
failure_target=π/4
dt = 0.1
Nsteps_per_episode = 20
noise_dist=Normal(0, 0.4)
Px, mdp = gen_topple_mdp(px=noise_dist, Nsteps=Nsteps_per_episode, dt=dt, failure_thresh=failure_target)
xs = Px.distribution.support
S = state_space(mdp)
A = action_space(Px)

# Networks
net(out_act...;Nout=A.N) = Chain(Dense(3, 32, relu), Dense(32, 32, relu), Dense(32, Nout, out_act...)) # Basic architecture
V() = ContinuousNetwork(net(sigmoid, Nout=1)) # Value network
Π() = DiscreteNetwork(net(), xs, always_stochastic=true) # Actor network (Policy gradient)
AC() = ActorCritic(Π(), V()) # Actor-crtic for baseline approaches
Q() = DiscreteNetwork(net(sigmoid, Nout=A.N), xs, always_stochastic=true, logit_conversion=estimator_logits(Px)) # Q network for value-based


# Solver parameters
Nbuff = Neps*Nsteps_per_episode
shared_params(name, π) = (agent=PolicyParams(;π, pa=Px), 
                          N=Neps, 
                          S,
                          f_target=failure_target,
                          buffer_size=Nbuff,
                          log=(dir="log/$name", period=1000))
						  
mc_params = (agent=PolicyParams(;π=Px, pa=Px), 
			 N=Neps, 
			 S, 
			 buffer_size=Nbuff, 
			 𝒫=(;f_target=failure_target))
			 
pg_params(name, π, use_baseline=false) = (ΔN=200, 
										  use_baseline,
										  agent_pretrain=pretrain_policy(mdp, Px),
										  shared_params(name, π)...)
vb_params(name, π) = (ΔN=4, 
					  N_elite_candidate=100, 
					  c_opt=(epochs=Nsteps_per_episode,), 
					  agent_pretrain=pretrain_value_discrete(mdp, Px, target=0.1),
					  shared_params(name, π)...)


## Get the ground truth comparison

# Ground truth experiment
# D = episodes!(Sampler(mdp, PolicyParams(π=Px, pa=Px)), explore=true, Neps = Neps)
# vals = D[:r][:][D[:done][:] .== 1]
# histogram(vals, label="Pendulum Topple", xlabel="Return", ylabel="Count", title="Distribution of Returns")
# sum(D[:r] .> π/4) / sum(D[:done][:] .== 1)

gt = 2.5333333333333334e-5
gt_std = 4.163331998932266e-6
plot_init = ()->plot(1:Neps, x->gt, linestyle=:dash, color=:black, ylims=(0,0.0001))

# Create our "run_experiment function"
run_experiment = experiment_setup(;mdp, Ntrials, dir, plot_init)

## Experiments: Compare a bunch of different variance reduction techniques
run_experiment(()->MCSolver(;mc_params...), "MC")
run_experiment(()->PolicyGradientIS(;pg_params("PG_nopretrain", Π())..., agent_pretrain=nothing), "PG_nopretrain")
run_experiment(()->PolicyGradientIS(;pg_params("PG", Π())...), "PG")
run_experiment(()->PolicyGradientIS(;pg_params("PG_baseline", AC(), true)...), "PG_baseline")
run_experiment(()->PolicyGradientIS(;pg_params("PG_defensive", MISPolicy([Π(), Px], [3, 1]))...), "PG_defensive")
run_experiment(()->PolicyGradientIS(;pg_params("PG_defensive_baseline", MISPolicy([AC(), Px], [3, 1]), true)...), "PG_defensive_baseline")
run_experiment(()->PolicyGradientIS(;pg_params("PG_MIS", MISPolicy([Π(), Π(), Π(), Px], [1, 1, 1, 1]))...), "PG_MIS")
run_experiment(()->ValueBasedIS(;vb_params("VB_nopretrain", Q())..., agent_pretrain=nothing),  "VB_nopretrain")
run_experiment(()->ValueBasedIS(;vb_params("VB", Q())...),  "VB")
run_experiment(()->ValueBasedIS(;vb_params("VB_defensive", MISPolicy([Q(), Px], [3, 1]))...), "VB_defensive")
run_experiment(()->ValueBasedIS(;vb_params("VB_MIS", MISPolicy([Q(), Q(), Q(), Px], [1, 1, 1, 1]))...),  "VB_MIS")

data = BSON.load("results/pendulum_discrete/VB_MIS.bson")[:data]

data[:fs]
mean(data[:ws])

i = 3
plot(cumsum(data[:fs][i] .* data[:ws][i]) ./ (1:100000))

𝒮vb = ValueBasedIS(;vb_params("VB_MIS", MISPolicy([Q(), Q(), Q(), Px], [1, 1, 1, 1]))...)
fs_vb, ws_vb = solve(𝒮vb, mdp)

nans = findall(isnan.(𝒮vb.buffer[:traj_importance_weight][1,:]))
ends = 𝒮vb.buffer[:episode_end][:,nans]
e1 = 1:findfirst(ends[:])
s = 𝒮vb.buffer[:s][:,nans[e1]]
a = 𝒮vb.buffer[:a][:,nans[e1]]
id = 𝒮vb.buffer[:id][:,nans[e1]]
D = Dict(:s=>s, :a=>a)

trajectory_pdf(𝒮vb.agent.π, D)

trajectory_pdf(𝒮vb.agent.π.distributions[1], D)
trajectory_pdf(𝒮vb.agent.π.distributions[2], D)
trajectory_pdf(𝒮vb.agent.π.distributions[3], D)
trajectory_pdf(𝒮vb.agent.π.distributions[4], D)

logpdf(𝒮vb.agent.π.distributions[3], D[:s], D[:a])

Crux.logits(𝒮vb.agent.π.distributions[3], D[:s])
Crux.value(𝒮vb.agent.π.distributions[3], D[:s])

vals = value(𝒮vb.agent.π.distributions[3], s)
probs = Float32.(Crux.logits(Px, s))
ps = vals .* probs
sm = sum(ps, dims=1)
zeros = sm[:] .== 0
sm[zeros] .= 1f0
ps[:, zeros] .= probs
ps
ps ./ sm

trajectory_pdf()

𝒮vbd = ValueBasedIS(;vb_params("VB_defensive", MISPolicy([Q(), Px], [3, 1]))...)
fs_vbd, ws_vbd = solve(𝒮vbd, mdp)



𝒮pg = PolicyGradientIS(;pg_params("PG", Π())...)
fs_pg, ws_pg = solve(𝒮pg, mdp)


plot(1:Neps, x->gt, linestyle=:dash, color=:black, ylims=(0,0.0001))
plot!(cumsum(fs_vb .* ws_vb) ./ (1:length(fs_vb)), label="VB")
plot!(cumsum(fs_vbd .* ws_vbd) ./ (1:length(fs_vbd)), label="vb_defensive")
plot!(cumsum(fs_pg .* ws_pg) ./ (1:length(fs_pg)), label="PG")

plot(log.(ws_vb), alpha=0.2, label="VB")
plot!(log.(ws_vbd), alpha=0.2, label="VB_defensive")
plot!(log.(ws_pg), alpha=0.2, label="PG")


data = BSON.load("results/pendulum_discrete/VB_defensive.bson")[:data]

plot(data[:ws][5])

data[:ws][5]

plot(1:Neps, x->gt, linestyle=:dash, color=:black, ylims=(0,0.0001))
plot!(cumsum(data[:fs][1][50000:end] .* data[:ws][1][50000:end]) ./ (1:length(data[:ws][1][50000:end])))

