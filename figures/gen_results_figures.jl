## Plot the trajectories
using Plots
pgfplotsx()
using LaTeXStrings
D1 = episodes!(Sampler(mdp, 𝒮.agent.π.distributions[1]), Neps=10)
D2 = episodes!(Sampler(mdp, 𝒮.agent.π.distributions[2]), Neps=10)

p = plot(xlabel="Time", ylabel="Angle", color_palette=:Dark2_3, xticks=[0,1,2], yticks=[-1,0,1], xlims=(0,2), ylims=(-1,1))
hline!([π/4, -π/4], color=:black, linestyle=:dash, label="Failure Threshold", legend=:topleft)
isfirst=true
for ep in episodes(ExperienceBuffer(D1))
	plot!(D1[:s][1,ep[1]:ep[2]], D1[:s][2,ep[1]:ep[2]], label=isfirst ? L"q_1" : "", alpha=0.3, color=2)
	isfirst=false
end
isfirst=true
for ep in episodes(ExperienceBuffer(D2))
	plot!(D2[:s][1,ep[1]:ep[2]], D2[:s][2,ep[1]:ep[2]], label=isfirst ? L"q_2" : "", alpha=0.3, color=3)
	isfirst=false
end
p
savefig("figures/mis.tex")


## Plot the values
pgfplotsx()
p = plot(xlabel="Time", ylabel="Angle", color_palette=:Dark2_3, xticks=[0,1,2], yticks=[-1,0,1], xlims=(0,2), ylims=(-1,1), )
heatmap!(0:0.1:2, -1.2:0.1:1.2, (t,θ) -> value(𝒮.agent.π.distributions[1], [t, θ, 0f0, 0f0])[1], clims=(0,1), color=:viridis)
savefig("figures/mode1.tex")
p = plot(xlabel="Time", ylabel="Angle", color_palette=:Dark2_3, xticks=[0,1,2], yticks=[-1,0,1], xlims=(0,2), ylims=(-1,1), )
heatmap!(0:0.1:2, -1.2:0.1:1.2, (t,θ) -> value(𝒮.agent.π.distributions[2], [t, θ, 0f0, 0f0])[1], clims=(0,1), color=:viridis)
savefig("figures/mode2.tex")
