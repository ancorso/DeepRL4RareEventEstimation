using Crux, Distributions, Plots
include("utils.jl")
include("../environments/mujoco_problem.jl")

init_mujoco_render() # Required for visualization

## Parameters
test = false

# Experiment params
if test
	Neps=500
	Ntrials=1
	Npretrain=1
else
	Neps=50_000
	Neps_gt=1_000_000
	Ntrials=10
	Npretrain=100
end
dir="results/halfcheetah"

# Problem setup params
dt = 0.1
Nsteps_per_episode = 50
xscale=0.04f0
failure_target=200f0

Px, mdp = gen_halfcheetah(;dt, Nsteps=Nsteps_per_episode, xscale)
S = state_space(mdp)
A = action_space(Px)

# Ground truth experiment
# D = episodes!(Sampler(mdp, PolicyParams(π=Px, pa=Px)), explore=true, Neps=1000)
# vals = D[:r][:][D[:done][:] .== 1]
# histogram(vals, label="Pendulum Topple", xlabel="Return", ylabel="Count", title="Distribution of Returns")
# sum(D[:r] .> π/4) / sum(D[:done][:] .== 1)
# 
# Crux.gif(mdp, Px, "out.gif", Neps=1, max_steps=50)

## Networks
function net(out_act...; Nin=S.dims[1], Nout=1, Nhiddens=[64, 32], act=tanh)
    hiddens = [Dense(Nhiddens[idx], Nhiddens[idx+1], relu) for idx in 1:length(Nhiddens)-1]
    Chain(Dense(Nin, Nhiddens[begin], act), hiddens..., Dense(Nhiddens[end], Nout, out_act...)) # Basic architecture
end

# Networks for on-policy algorithms
function Π(Nhiddens=[64, 32])
	base = net(Nout=32, Nhiddens=Nhiddens)
	μ = ContinuousNetwork(Chain(base..., Dense(32, A.dims[1])))
	logΣ = ContinuousNetwork(Chain(base..., Dense(32, A.dims[1])))
	GaussianPolicy(μ, logΣ, true)
end
V(Nhiddens=[64, 32]) = ContinuousNetwork(net(sigmoid; Nhiddens))
AC(Nhiddens=[64, 32]) = ActorCritic(Π(Nhiddens), V(Nhiddens))

# Networks for off-policy algorithms
QSA(Nhiddens=[256,256]) = ContinuousNetwork(net(sigmoid, Nin=S.dims[1] + A.dims[1], Nhiddens=Nhiddens, act=relu))
AQ(Nhiddens=[256,256]) = ActorCritic(Π(Nhiddens), QSA(Nhiddens))

function Π_CEM(Npolicies; d=()->product_distribution([Normal(rand(Distributions.Uniform(-0.1, 0.1)), 1.0) for _=1:A.dims[1]]))
	if Npolicies == 1
		return DistributionPolicy(d())
	else
		return MISPolicy([DistributionPolicy(d()) for _ in 1:Npolicies], trainable_indices=collect(1:Npolicies))
	end
end


## Algorithm parameter setup
Nbuff = Neps*Nsteps_per_episode
shared_params(name, π) = (
	agent=PolicyParams(;π, pa=Px), 
    N=Neps, 
    S,
    f_target=failure_target,
    buffer_size=Nbuff,
    log=(dir="log/$name", period=1000)
)
						  
mc_params(N=Neps) = (
	agent=PolicyParams(;π=Px, pa=Px), 
	N=N, 
	S, 
	buffer_size=N*Nsteps_per_episode, 
	𝒫=(;f_target=failure_target)
)
			 
cem_params(name, π) = (
	ΔN=200, 
	training_buffer_size=200*Nsteps_per_episode, 
	shared_params(name, π)...
)
			 
pg_params(name, π; use_baseline=false, kwargs...) = (
	ΔN=200, 
	use_baseline,
	a_opt=(;optimizer=Flux.Optimiser(Flux.ClipValue(1f0), Adam(3f-4)), batch_size=1024),
	c_opt=(;max_batches=200, batch_size=1024),
	training_buffer_size=200*Nsteps_per_episode,
	agent_pretrain=use_baseline ? pretrain_AV(mdp, Px, v_target=0.1, Nepochs=Npretrain) : pretrain_policy(mdp, Px, Nepochs=Npretrain),
	shared_params(name, π)...,
	kwargs...
)

vb_params(name, π; kwargs...) = (
	ΔN=20,
	train_actor=true,
	training_buffer_size=3200*Nsteps_per_episode,
	c_opt=(epochs=20,), 
	agent_pretrain=pretrain_AQ(mdp, Px, v_target=0.1, Nepochs=Npretrain),
	shared_params(name, π)...,
	kwargs...
)

## Ground truth
# data = experiment_setup(;mdp, Ntrials, dir)(()->MCSolver(;mc_params(Neps_gt)...), "gt")
# gt = mean(data[:est])[end]
# gt_std = std(data[:est])[end]
# println("gt: $gt, std: $std")

# gt = mean(data[:est])[end]
# gt_std = std(data[:est])[end]

plot(1:Neps, x->gt, linestyle=:dash, color=:black, ylims=(0,0.0001))

𝒮 = CEMSolver(;cem_params("CEM_1", Π_CEM(1))...)
fs, ws = solve(𝒮, mdp)
plot!(cumsum(fs .* ws) ./ (1:length(ws)), label="CEM1")

𝒮 = CEMSolver(;cem_params("CEM_2", Π_CEM(2))...)
fs, ws = solve(𝒮, mdp)
plot!(cumsum(fs .* ws) ./ (1:length(ws)), label="CEM2")

𝒮 = PolicyGradientIS(;pg_params("PG", Π())...)
fs, ws = solve(𝒮, mdp)
plot!(cumsum(fs .* ws) ./ (1:length(ws)), label="PG")

𝒮 = PolicyGradientIS(;pg_params("PG_MIS2", MISPolicy([Π(), Π()]))...)
fs, ws = solve(𝒮, mdp)
plot!(cumsum(fs .* ws) ./ (1:length(ws)), label="PG_MIS2")

𝒮 = ValueBasedIS(;vb_params("VB", AQ())...)
fs, ws = solve(𝒮, mdp)
plot!(cumsum(fs .* ws) ./ (1:length(ws)), label="VB")

𝒮 = ValueBasedIS(;vb_params("VB_MIS2", MISPolicy([AQ(), AQ()]))...)
fs, ws = solve(𝒮, mdp)
plot!(cumsum(fs .* ws) ./ (1:length(ws)), label="VB_MIS2")








if test
	for (𝒮fn, name) in standard_exps; run_experiment(𝒮fn, name); end
else
	Threads.@threads for (𝒮fn, name) in shuffle(standard_exps); run_experiment(𝒮fn, name); end
end




