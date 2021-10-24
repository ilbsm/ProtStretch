from Trajectory import Trajectory
from Tools import *
import matplotlib.pyplot as plt
from copy import deepcopy


class Experiment(Trajectory):

    def __init__(self, filename, case, bond_length, residues, initial_guess, bounds, **kwargs):

#         bond_length = 0.365
#         residues = 226
        # residues = 1288
        # residues = 188-14
        # initial_guess = [5, 0]
        # bounds = ((4.8, 5.2), (0, 0))
#         initial_guess = [3, 0.001]
#         bounds = ((1, 5), (0.0001, 0.003))
        # initial_guess = [4, 0]
        # bounds = ((3, 7), (0, 0))
        # self._peaks = pd.DataFrame({'heights': [0.1, 0.1, 0.1, 0.1], 'means': [18, 27.5, 52, 67], 'widths': [1, 1, 1,
        #                                                                                                     1]}) #tm15
        self._peaks = pd.DataFrame({'heights': [0.1, 0.1, 0.1], 'means': [32, 45, 81], 'widths': [1, 1, 1]}) #trmd
        # self._peaks = pd.DataFrame({'heights': [0.1, 0.1], 'means': [410, 480], 'widths': [10, 10]})

        super().__init__(filename, case, bond_length, residues, initial_guess, bounds, **kwargs)

    def _find_peaks(self):
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, "p_k_table/")

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        with open(results_dir + "p_k_results_all.txt") as myfile:

            if ("trace" + str(self._case)) in myfile.read():

                csv_data = pd.read_table(results_dir + "p_k_results_all.txt", delim_whitespace=True, header=0)
                self.p = csv_data.loc[csv_data["trace"] == "trace" + str(self._case)]["p"].values[0]
                self.k = csv_data.loc[csv_data["trace"] == "trace" + str(self._case)]["k"].values[0]
            else:
                myfile.close()
            if not (hasattr(self, "p") or hasattr(self, "k")):
                fitted = self._fit(self._find_last_range())
                self.p = fitted[0]
                self.k = fitted[1]
                # self.p = 6
                # self.k = 3.69180877e-04
                # self.k = 0.2e-03
                self.parameters_to_txt("p_k_results_all")

        L = find_contour_lengths(self._data, self.p, self.k)
        self._data["L"] = L
        histo_data = decompose_histogram(self._data["L"], guess=self._peaks)

        return histo_data

    def _to_minimize(self, x, last_range):
        print(x)
        fit_data = self._data[self._data['d'].between(last_range[0], last_range[1])]
        fit_data = fit_data.reset_index(drop=True)
        fit_f = wlc(fit_data['d'], self.bond_length * (self.residues - 1), x[0], x[1])
        return np.linalg.norm(fit_f - fit_data['F'].to_numpy())

    def _fit(self, last_range):
        opt = minimize(self._to_minimize, x0=self.initial_guess, args=list(last_range), method='TNC',
                       bounds=self.bounds)
        return opt['x']

    def state_boundaries(self):

        """Finding boundaries of each state. A boundary is defined when WLC fit for each state reaches the value 12.
        Method changes histo_data field - it adds two new columns: 'begs' and 'ends'. Returns None.

        """

        begs = [round(self._smooth_data['d'].min(), 3)]
        ends = []
        for mean in self.histo_data['means']:
            cut = invert_wlc(25, self.p, self.k) * mean
            bound = self._smooth_data.iloc[(self._smooth_data['d'] - cut).abs().argsort()[:1]]['d'].to_list()[0]
            data_near = self._smooth_data.loc[self._smooth_data['F']
                                              == self._smooth_data.loc[abs(self._smooth_data['d'] - bound)
                                                                       < 1]['F'].max()]['d'].min()

            # if float(43) < mean < float(45):
            #     data_near = self._smooth_data.loc[self._smooth_data['F']
            #                                       == self._smooth_data.loc[abs(self._smooth_data['d'] - bound)
            #                                                                < 3]['F'].max()]['d'].min()

            data_near = round(data_near, 3)
            if mean == self._histo_data['means'].iloc[-1]:
                ends.append(self._smooth_data['d'].iloc[-1])
                break
            begs.append(data_near)
            ends.append(data_near)

        # boundaries = [[begs[i], ends[i]] for i in range(len(begs))]
        self.histo_data['begs'] = begs
        self.histo_data['ends'] = ends

    def plot_fd(self, position):

        position.plot(self._data.sort_values(by='d')['d'], self._data.sort_values(by='d')['F'])

        if hasattr(self, "_data_inverse"):
            position.plot(self._data_inverse.sort_values(by='d')["d"], self._data_inverse.sort_values(by='d')['F'],
                          color=mcolors.CSS4_COLORS['lightsteelblue'])
        position.plot(self._smooth_data['d'], self._smooth_data['F'], color=mcolors.CSS4_COLORS['pink'])
        index = 0
        for mean in self.histo_data['means']:
            residues = 1 + int(mean / self.bond_length)
            d_space = np.linspace(1, mean)
            # self._data['d'].max()

            label = "L= " + str(round(mean, 3)) + ' (' + str(residues) + ' AA)'
            y_fit = wlc(d_space, mean, self.p, self.k)

            position.plot(d_space, y_fit, ls='--', linewidth=1, label=label, color=get_color(index))
            index += 1

        position.set_ylim(0, 45)
        position.set_xlim(min(self._data['d']), max(self._data['d']))
        position.set_title('Trace fits')
        position.set_xlabel('Extension [nm]')
        position.set_ylabel('Force [pN]')
        position.legend(fontsize='small')
