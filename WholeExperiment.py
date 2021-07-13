from Whole import Whole
from Experiment import Experiment
from Tools import dudko_hummer_szabo


class WholeExperiment(Whole):

    def __init__(self, filename, number_of_traces, **kwargs):
        self._kwargs = kwargs
        self._filename = filename
        self._number_of_traces = number_of_traces
        super().__init__(number_of_traces)

    def _collect_traces(self):
        return [Experiment(self._filename, i, **self._kwargs) for i in range(1, self._number_of_traces)]

    def _get_total_trace(self):
        return Experiment("total_exp.txt", case=None, p=self.p_mean, k=self.k_mean)

    def _analyze_total_trace(self):
        self._total_trace.state_boundaries()
        self._total_trace.plot(name="total_exp")
        print(self._total_trace.histo_data)

    def dudko(self):
        dhs_table = dudko_hummer_szabo(self.rupture_table, self.states_rup)
        return dhs_table




