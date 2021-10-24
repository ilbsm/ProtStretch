from Theory import Theory
from Experiment import Experiment
from WholeExperiment import WholeExperiment
from WholeTheory import WholeTheory

# PARAMETRY EKSPERYMENTU
p_dna = 0.16
l_dna = 350
k_dna = 0.003

#(ODKOMENTUJ TYLKO JEDEN Z PONIŻSZYCH BLOKÓW)
# uzupełnij to na dole:
# bond_length =
# residues =  
# initial_guess =
# bounds = 
#JAK ANALIZOWAĆ POJEDYNCZĄ TRAJEKTORIĘ Z SYMULACJI 
trace = Theory("aa01.afm", 1, bond_length, residues, initial_guess, bounds)
trace.state_boundaries()
trace.plot()
trace.calculate_work()
trace.rupture_forces()
print(trace.histo_data)

#JAK ANALIZOWAĆ POJEDYNCZĄ TRAJEKTORIĘ Z EKSPERYMENTU

trace = Experiment("data_test.xls", 1, bond_length, residues, initial_guess, bounds, p_dna=p_dna, l_dna=l_dna, k_dna=k_dna)
trace.state_boundaries()
trace.plot()
trace.state_boundaries()
trace.calculate_work()
trace.rupture_forces()
print(trace.histo_data)


#JAK ANALIZOWAĆ ZESTAW TRAJEKTORII Z EKSPERYMENTU
traces = WholeExperiment("data_test.xls", 21, p_dna=p_dna, l_dna=l_dna, k_dna=k_dna)
traces.draw_rupture_histo()
print(traces.rupture_table)
traces.plot_all()
res = traces.crooks()
print(res)


#JAK ANALIZOWAĆ ZESTAW TRAJEKTORII Z SYMULACJI
traces = WholeTheory('aa', 21)
traces.plot_all()
print(traces.rupture_table)
traces.draw_rupture_histo()
print(traces.rupture_table)
res = traces.crooks()
print(res)
