from Trajectory import Trajectory
from Tools import *


class Theory(Trajectory):

    def __init__(self, filename, case, **kwargs):

        bond_length = 0.38
        # residues = 240 #trmd
        residues = 188 #tm15
        initial_guess = [4, 9.67e-05]
        bounds = ((3, 5), (0.00001, 0.001))
        
        # określ słownik z parametrami pików na histogramie CL

        # self._peaks = pd.DataFrame({'heights': [0.1, 0.1, 0.1, 0.1, 0.1], 'means': [7.5, 31, 46, 77, 90],
        #                             'widths': [1, 1, 1, 1, 1]}) # TRMD.PDB
        # self._peaks = pd.DataFrame({'heights': [0.1, 0.1, 0.1, 0.1], 'means': [5, 20, 57, 70],
        #                                                          'widths': [1, 1, 1, 1]}) # tmx2
        self._peaks = pd.DataFrame({'heights': [0.1, 0.1, 0.1, 0.1, 0.2], 'means': [5, 16, 22, 57, 70],
                                    'widths': [1, 1, 1, 1, 1]})  # 3dcm
        # self._peaks = pd.DataFrame({'heights': [0.1, 0.1, 0.1, 0.1, 0.1], 'means': [7.5, 31, 46, 77, 90],
        #                             'widths': [1, 1, 1, 1, 1]}) # 5wyr

        super().__init__(filename, case, bond_length, residues, initial_guess, bounds, **kwargs)

    def _find_peaks(self):
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, "p_k_table/")

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        with open(results_dir + "p_k_results.txt") as myfile:

            if ("trace" + str(self._case)) in myfile.read():
                csv_data = pd.read_table(results_dir + "p_k_results.txt", delim_whitespace=True, header=0)
                self.p = csv_data.loc[csv_data["trace"] == "trace" + str(self._case)]["p"].values[0]
                self.k = csv_data.loc[csv_data["trace"] == "trace" + str(self._case)]["k"].values[0]
            else:
                myfile.close()
                if not (hasattr(self, "p") or hasattr(self, "k")):
                    fitted = self._fit(self._find_last_range())
                    self.p = fitted[0]
                    self.k = fitted[1]
                    self.parameters_to_txt("p_k_results")

        L = find_contour_lengths(self._data, self.p, self.k)
        self._data["L"] = L
        histo_data = decompose_histogram(self._data["L"], guess=self._peaks)

        return histo_data

    def _to_minimize(self, x, last_range):
        print(x)
        fit_data = self._data[self._data['d'].between(last_range[0], last_range[1])]
        length = self.bond_length * (self.residues - 1)
        fit_f = wlc(fit_data['d'], length, x[0], x[1])
        return np.linalg.norm(fit_f - fit_data['F'].to_numpy())

    def _fit(self, last_range):
        opt = minimize(self._to_minimize, x0=self.initial_guess, args=list(last_range), method='TNC',
                       bounds=self.bounds)
        return opt['x']

    def state_boundaries(self):
        begs = [round(self._smooth_data['d'].min(), 3)]
        ends = []

        for mean in self.histo_data['means']:
            cut = invert_wlc(80, self.p, self.k) * mean
            bound = self._smooth_data.iloc[(self._smooth_data['d'] - cut).abs().argsort()[:1]]['d'].to_list()[0]
            data_near = self._smooth_data.loc[self._smooth_data['F']
                                              == self._smooth_data.loc[abs(self._smooth_data['d'] - bound)
                                                                       < 2]['F'].max()]['d'].min()
            # dealing with 72 boundary
            if float(70) < mean < float(74):

                try:
                    maximas = argrelextrema(
                        self._smooth_data.loc[abs(self._smooth_data['d'] - bound) < 5]['F'].to_numpy(),
                        np.greater)
                    data_near = self._smooth_data.loc[abs(self._smooth_data['d'] - bound) < 5]['d'].to_numpy()[
                        maximas[0].min()]
                except:
                    data_near = self._smooth_data.loc[self._smooth_data['F']
                                                      == self._smooth_data.loc[abs(self._smooth_data['d'] - bound)
                                                                               < 2]['F'].max()]['d'].min()

            # dealing with the wide one
            if float(30) < mean < float(35):
                data_near = self._smooth_data.loc[self._smooth_data['F']
                                                  == self._smooth_data.loc[abs(self._smooth_data['d'] - bound)
                                                                           < 5]['F'].max()]['d'].min()

            data_near = round(data_near, 3)
            if mean == self._histo_data['means'].iloc[-1]:
                ends.append(data_near)
                break
            begs.append(data_near)
            ends.append(data_near)

        # boundaries = [[begs[i], ends[i]] for i in range(len(begs))]
        self._histo_data['begs'] = begs
        self._histo_data['ends'] = ends

    def plot_fd(self, position):

        # fig = plt.figure(dpi = 200, figsize = (5, 5))
        # plt.plot(self._data.sort_values(by='d')['d'], self._data.sort_values(by='d')['F'])
        # plt.title('Examplary stretching curve')
        # plt.xlabel('Extension')
        # plt.ylabel('Force')
        # plt.show()

        if hasattr(self, "_data_inverse"):
            position.plot(self._data_inverse.sort_values(by='d')["d"], self._data_inverse.sort_values(by='d')['F'],
                          color=mcolors.CSS4_COLORS['lightsteelblue'],  label='Refolding')
        position.plot(self._data.sort_values(by='d')['d'], self._data.sort_values(by='d')['F'], label='Folding')
        position.plot(self._smooth_data['d'], self._smooth_data['F'], color=mcolors.CSS4_COLORS['pink'])
        index = 0
        for mean in self.histo_data['means']:
            residues = 1 + int(mean / self.bond_length)
            d_space = np.linspace(1, self._data['d'].max())

            label = "L= " + str(round(mean, 3)) + ' (' + str(residues) + ' AA)'
            y_fit = wlc(d_space, mean, self.p, self.k)

            position.plot(d_space, y_fit, ls='--', linewidth=1, label=label, color=get_color(index))
            index += 1

        position.set_ylim(0, 120)
        position.set_xlim(min(self._data['d']), max(self._data['d']))
        position.set_title('F(d) curve for folding and refolding of 3DCM')
        position.set_xlabel('Extension [nm]')
        position.set_ylabel(r'Force [$\frac{\epsilon}{nm}$]')
        position.legend(fontsize='small')

