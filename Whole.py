import pandas as pd
from Tools import dudko_hummer_szabo, get_color
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


class Whole:
    
    DIRMANE="data_3dcm/"

    def __init__(self, number_of_traces):

        if number_of_traces == 1:
            raise ValueError

        self._traces = self._collect_traces()
        self._compute_all()
        self.p_mean = sum([trace.p for trace in self._traces])/number_of_traces
        self.k_mean = sum([trace.k for trace in self._traces]) / number_of_traces
        self._data_total = pd.concat([trace.data for trace in self._traces], ignore_index=True)
        self._data_total.to_csv(self.DIRNAME+"total.txt", header=True, index=False, sep=' ', mode='w')
        self._total_trace = self._get_total_trace()
        self._analyze_total_trace()

        self.states, self.states_rup, self.states_i = self._trajectory_loop()
        self._save_ruptures()
        self._save_works()
        self.rupture_table = pd.DataFrame({})

    def draw_rupture_histo(self):
        font = {'family': 'Arial',
                'weight': 'normal',
                'size': 12}
        matplotlib.rc('font', **font)

        script_dir = os.path.dirname(__file__)
        results_dir_rup = os.path.join(script_dir, 'Force_histograms/')
        plt.figure(figsize=(5, 5))
        param_list = []
        for i, state in enumerate(self.states_rup):
            if i == len(self.states_rup)-1:
                break
            # Fit a normal distribution to the data:
            mu, std = norm.fit(self.states_rup[state])
            # Plot the histogram.
            # Plot the PDF.
            f_space = np.linspace(0, 110, 1000)
            # bylo range do 20 i 130 binow
            # lub do 47 i 50 binow // 80 range
            p = norm.pdf(f_space, mu, std)
            plt.plot(f_space, p, ls='--', linewidth=0.5, color=get_color(i))
            if i != len(self.states_rup)-1:
                label = "state "+str(i+1)
            plt.hist(self.states_rup[state], bins=50, density=True, alpha=0.3, color=get_color(i), range=[0, 110],
                     label=label)
            params = pd.DataFrame({'heights': [max(p)], 'means': [mu], 'widths': [std]})
            param_list.append(params)

        self.rupture_table = pd.concat([par for par in param_list], ignore_index=True)
        # plt.xlabel(r"Rupture force [$\frac{\epsilon}{nm}$]")
        plt.xlabel("Rupture force [pN]")
        plt.ylabel("Probability")
        plt.title("Rupture forces histogram")
        plt.legend(fontsize='small')
        plt.tight_layout()
        sample_file_name = 'Force_histo'
        plt.savefig(results_dir_rup + sample_file_name)
        plt.close()

    def dudko(self):
        raise NotImplemented

    def _save_ruptures(self):
        script_dir = os.path.dirname(__file__)
        results_dir_rup = os.path.join(script_dir, 'Force_histograms/')
        if not os.path.isdir(results_dir_rup):
            os.makedirs(results_dir_rup)

        text_file = open("Force_histograms/forces" + str(len(self._traces)) + ".txt", "w")
        text_file.write(str(self.states_rup))
        text_file.close()

    def _save_works(self):
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'Work_histograms/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        text_file = open("Work_histograms/works" + str(len(self._traces)) + ".txt", "w")
        text_file.write(str(self.states))
        text_file.close()

    def _get_total_trace(self):
        raise NotImplemented

    def _analyze_total_trace(self):
        raise NotImplemented

    def _collect_traces(self):
        raise NotImplemented()

    def _compute_all(self):
        for trace in range(len(self._traces)):
            self._traces[trace].state_boundaries()
            self._traces[trace].calculate_work()
            self._traces[trace].rupture_forces()

    def plot_all(self):
        for trace in range(len(self._traces)):
            self._traces[trace].plot()

    def save_all(self):
        for trace in range(len(self._traces)):
            self._traces[trace].results_to_latex(self._traces[trace].histo_data)
            self._traces[trace].results_to_txt()

    def _trajectory_loop(self, tolerance=3, start=0):

        """Classifies states of one trace to states identified in total histogram.

        :param start: Starting trace
        :type start: int
        :param tolerance: max difference between states boundaries
        :type tolerance: int/float
        :return: dictionary (keys: states, values: works)
        :rtype: dict
        """

        dct1 = {}
        dct2 = {}
        dct3 = {}

        for i in range(len(self._total_trace.histo_data)):
            dct1['state' + str(i)] = []
            dct2['state' + str(i)] = []
            dct3['state' + str(i)] = []

        for trace in self._traces[start:-1]:
            for index, row in self._total_trace.histo_data.iterrows():
                if index == len(trace.histo_data):
                    break

                if abs(row['ends'] - trace.histo_data['ends'][index]) < tolerance:
                    dct1['state' + str(index)].append(trace.histo_data['work-s'][index])
                    dct2['state' + str(index)].append(trace.histo_data['rupture'][index])
                    dct3['state' + str(index)].append(trace.histo_data['work-i'][index])
                else:
                    for index2, row2 in self._total_trace.histo_data[index + 1:].iterrows():
                        if abs(row2['begs'] - trace.histo_data['begs'][index]) < tolerance:
                            if abs(row2['ends'] - trace.histo_data['ends'][index]) < tolerance:
                                dct1['state' + str(index2)].append(trace.histo_data['work-s'][index])
                                dct2['state' + str(index2)].append(trace.histo_data['rupture'][index])
                                dct3['state' + str(index2)].append(trace.histo_data['work-i'][index])
                                break

        return dct1, dct2, dct3

    def _draw_work_histo(self):
        """ Plotting works histograms and saving them to Images/Work_histograms
                """

        font = {'family': 'Arial',
                'weight': 'normal',
                'size': 12}
        matplotlib.rc('font', **font)

        script_dir = os.path.dirname(__file__)
        results_dir_work = os.path.join(script_dir, 'Work_histograms/')

        params_refolding = []
        params_folding = []
        i=1
        for state, state_i in zip(self.states, self.states_i):
            w_space = np.linspace(0, max(self.states[state]) + 50, 1000)
            plt.figure(figsize=(5, 5))
            # Fit a normal distribution to the data:
            mu, std = norm.fit(self.states[state])
            params_refolding.append([mu, std])
            mu2, std2 = norm.fit(self.states_i[state])
            params_folding.append([mu2, std2])
            # Plot the histogram.
            plt.hist(self.states[state], bins=40, density=True, alpha=0.3, color='r', range=[0, max(self.states[
                                                                                                        state])],
                     label="Refolding")
            plt.hist(self.states_i[state], bins=40, density=True, alpha=0.3, color='b', range=[0, max(self.states[
                                                                                                         state])],
                     label="Folding")
            # Plot the PDF.
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax+100, 1000)
            p = norm.pdf(x, mu, std)
            p2 = norm.pdf(x, mu2, std2)
            plt.plot(x, p, '--r', linewidth=1)
            plt.plot(x, p2, '--b', linewidth=1)
            # title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
            title = "State " + str(i)
            i+=1
            print("mu = %.2f,  std = %.2f" % (mu, std))
            print("mu = %.2f,  std = %.2f" % (mu2, std2))
            plt.title(title)
            plt.xlabel(r"Work [$\epsilon$]")
            # plt.xlabel(r"Work [$pN \cdot nm$]")
            plt.ylabel('Probability')
            plt.legend(fontsize='small')
            plt.tight_layout()
            sample_file_name = state
            plt.savefig(results_dir_work + sample_file_name)
            plt.close()
        return params_refolding, params_folding

    def crooks(self):
        params_refolding, params_folding = self._draw_work_histo()

        def solve(m1, m2, std1, std2):
            a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
            b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
            c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)
            return np.roots([a, b, c])

        results = []

        for re, fo in zip(params_refolding, params_folding):
            zeros = solve(re[0], fo[0], re[1], fo[1])
            results.append(zeros)

        print(results)

        return results
