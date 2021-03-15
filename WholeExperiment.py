from Whole import Whole
from Experiment import Experiment
from Tools import decompose_histogram


class WholeExperiment(Whole):

    def __init__(self, filename, number_of_traces, **kwargs):
        self._kwargs = kwargs
        self._filename = filename
        self._number_of_traces = number_of_traces
        super().__init__(number_of_traces)
        self._total_trace = Experiment("total_exp.txt", case=None, p=self.p_mean, k=self.k_mean)
        self._total_trace.state_boundaries()
        self._total_trace.plot(name="total_exp")
        print(self._total_trace.histo_data)

    def _collect_traces(self):
        return [Experiment(self._filename, i, **self._kwargs) for i in range(1, self._number_of_traces)]



