using POMDPGym, Crux, Distributions, POMDPs, Flux, JLD2

function gen_halfcheetah(;dt=0.1f0, Nsteps=50, xscale=0.05f0, maxR=200f0)
   maxT = (Nsteps-1)*dt

   # Build the MDP
   mdp = GymPOMDP(:HalfCheetah, version = :v3)
   S = state_space(mdp)
   adim = length(POMDPs.actions(mdp)[1])

   # Load the policy
   policy = load("environments/half_cheetah_policy.jld2")["policy"]
   policy.network = Chain(policy.network..., (x) -> x .* 0.9f0) # for some reason jld2 dropped anon function

   # Construct the Mujoco environment
   px = product_distribution([Normal(0, 1f0) for _=1:adim])
   Px = DistributionPolicy(px)

   cost_fn(m, s, sp) = isterminal(m, sp) ? maxR - (s[2] / 0.02f0) : 0

   rmdp = RMDP(mdp, policy; cost_fn, added_states=:time_and_acc_reward, dt, maxT, xscale, disturbance_type=:action_noise, rshift=0f0, rscale=0.02f0)
   Px, rmdp
end


