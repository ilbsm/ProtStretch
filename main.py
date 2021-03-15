from Theory import Theory
from Experiment import Experiment
from WholeExperiment import WholeExperiment

# trace = Theory("aa01.afm", 1)
# trace.plot()
# trace.state_boundaries()
# trace.calculate_work()
# trace.rupture_forces()
# print(trace.histo_data)

p_dna = 0.16
l_dna = 350
k_dna = 0.003
# trace = Experiment("data_test.xls", 1, p_dna=p_dna, l_dna=l_dna, k_dna=k_dna)
# trace.state_boundaries()
# trace.plot()
# # trace.state_boundaries()
# trace.calculate_work()
# trace.rupture_forces()
# print(trace.histo_data)

traces = WholeExperiment("data_test.xls", 10, p_dna=p_dna, l_dna=l_dna, k_dna=k_dna)
traces.plot_all()