using Printf

function convergence_plot(filepath, gt, Nsamps)
	data = BSON.load(filepath)[:data]
	Ntrial = length(data[:est])

	p = plot(1:Nsamps, x->gt, linestyle=:dash, color=:black, ylims=(0,gt*5))
	plot!(1:Nsamps, mean(data[:est]), ribbon=std(data[:est]))
end

function abs_rel_err(filepath, gt, Nsamps)
	data = BSON.load(filepath)[:data]
	Ntrial = length(data[:est])
	
	rel_errs = [(data[:est][i][Nsamps] - gt)/gt for i=1:Ntrial]
	
	mean_abs_rel_err = mean(abs.(rel_errs))
	std_abs_rel_err = std(abs.(rel_errs))
	@sprintf("\$%.2f \\pm %.2f\$", mean_abs_rel_err, std_abs_rel_err)
end

function rel_err(filepath, gt, Nsamps)
	data = BSON.load(filepath)[:data]
	Ntrial = length(data[:est])
	
	rel_errs = [(data[:est][i][Nsamps] - gt)/gt for i=1:Ntrial]
	
	mean_rel_err = mean(rel_errs)
	std_rel_err = std(rel_errs)
	@sprintf("\$%.2f \\pm %.2f\$", mean_rel_err, std_rel_err)
end

environments = [
	("\\multicolumn{2}{c}{\\textbf{Pendulum (Discrete)}}", "external_results/updated/pendulum_discrete/", 50000, 2.5333333333333334e-5),
	("\\multicolumn{2}{c}{\\textbf{Pendulum (Continuous)}}", "external_results/updated/pendulum_continuous/", 50000, 1.96f-5),
]


baseline_experiment = [
	("PG", ["PG.bson", "PG_baseline.bson"]),
	("PG MIS-2", ["PG_MIS2.bson", "PG_MIS2_baseline.bson"]),
	("PG MIS-4", ["PG_MIS4.bson", "PG_MIS4_baseline.bson"]),
]
baseline_subheaders = ["\\textbf{Method}", "\\textbf{No Baseline}", "\\textbf{Baseline}",  "\\textbf{No Baseline}", "\\textbf{Baseline}"]


pretrain_experiment = [
	("PG", ["PG_nopretrain.bson", "PG.bson"]),
	("PG MIS-2", ["PG_MIS2_nopretrain.bson","PG_MIS2.bson"]),
	("PG MIS-4", ["PG_MIS4_nopretrain.bson","PG_MIS4.bson"]),
	("VB", ["VB_nopretrain.bson","VB.bson"]),
	("VB MIS-2", ["VB_MIS2_nopretrain.bson","VB_MIS2.bson"]),
	("VB MIS-4", ["VB_MIS4_nopretrain.bson","VB_MIS4.bson"])
]
pretrain_subheaders = ["\\textbf{Method}", "\\textbf{No Pretrain}", "\\textbf{Pretrain}",  "\\textbf{No Pretrain}", "\\textbf{Pretrain}"]

defensive_experiment = [
	("PG", ["PG.bson", "PG_defensive.bson"]),
	("PG MIS-2", ["PG_MIS2.bson", "PG_MIS2_defense.bson"]),
	("PG MIS-4", ["PG_MIS4.bson", "PG_MIS4_defense.bson"]),
	("VB", ["VB.bson", "VB_defensive.bson"]),
	("VB MIS-2", ["VB_MIS2.bson", "VB_MIS2_defensive.bson"]),
	("VB MIS-4", ["VB_MIS4.bson", "VB_MIS4_defensive.bson"])
]
defensive_subheaders = ["\\textbf{Method}", "\\textbf{Vanilla}", "\\textbf{Defensive}",  "\\textbf{Vanilla}", "\\textbf{Defensive}"]


Npolicies_experiment = [
	("MC", ["MC.bson", "MC.bson", "MC.bson"]),
	("CEM", ["CEM_1.bson", "CEM_2.bson", "CEM_4.bson"]),
	("PG", ["PG.bson", "PG_MIS2.bson", "PG_MIS4.bson"]),
	("VB", ["VB.bson", "VB_MIS2.bson", "VB_MIS4.bson"]),
]
Npolicies_subheaders = ["\\textbf{Method}", "\\textbf{\$M=1\$}", "\\textbf{\$M=2\$}",  "\\textbf{\$M=4\$}", "\\textbf{\$M=1\$}", "\\textbf{\$M=2\$}",  "\\textbf{\$M=4\$}",]




function gen_figures_and_tables(environments, experiments, metrics, subheaders, output_file)
	io = open(output_file, "w")
	write(io, "\\toprule\n")
	for (name, _, _, _) in environments
		write(io, string(" & ", name))
	end
	write(io, "\\\\ \n")
	write(io, "\\midrule\n")
	write(io, subheaders[1])
	
	for h in subheaders[2:end]
		write(io, string(" & ", h))
	end
	write(io, "\\\\\n\\midrule\n")
	
	plotcolumns=0
	plotrows=0
	plots = []
	
	for (method, files) in experiments
		plotrows += 1
		plotcolumns=0
		write(io, method)
		for (name, dir, Nsamps, gt) in environments
			for f in files
				path = string(dir, f)
				push!(plots, convergence_plot(path, gt, Nsamps))
				plotcolumns += 1
				for m in metrics
					write(io, string(" & ", m(path, gt, Nsamps)))
				end
			end
		end
		write(io, "\\\\ \n")
	end
	write(io, "\\bottomrule\n")
	close(io)
	plot(plots..., layout=(plotrows, plotcolumns), size=(300, 200).*((plotcolumns, plotrows)))
end

gen_figures_and_tables(environments, pretrain_experiment, [abs_rel_err], pretrain_subheaders, "pretrain_experiment.txt")
gen_figures_and_tables(environments, defensive_experiment, [abs_rel_err], defensive_subheaders, "defensive_experiment.txt")
gen_figures_and_tables(environments, baseline_experiment, [abs_rel_err], baseline_subheaders, "baseline_experiment.txt")
gen_figures_and_tables(environments, Npolicies_experiment, [abs_rel_err], Npolicies_subheaders, "Npolicies_accuracy_experiment.txt")
gen_figures_and_tables(environments, Npolicies_experiment, [rel_err], Npolicies_subheaders, "Npolicies_bias_experiment.txt")


