import pandas as pd
from io import StringIO
import numpy as np
from numpy.core._multiarray_umath import ndarray
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit, minimize
from sklearn.neighbors import KernelDensity
import matplotlib.colors as mcolors
from scipy.integrate import simps
import os
from scipy.special import erf
import matplotlib.pyplot as plt
from scipy.stats import norm
from random import uniform


def load_data(path, inverse=False):
    """ Method of loading the data of the trace.

                    :param path: path
                    :type path: str

                    :return: loaded data
                    :rtype:  DataFrame

                    """

    if path == os.path.join(os.path.dirname(__file__), 'data/total.txt'):
        data = pd.read_table("data/total.txt", delim_whitespace=True, header=0)
        return data

    if path == os.path.join(os.path.dirname(__file__), 'data_exp_tm1570/total.txt'):
        data = pd.read_table("data_exp_tm1570/total.txt", delim_whitespace=True, header=0)
        return data

    with open(path, 'r') as file:
        content = file.read().split("#")[-1].strip()

    data = pd.read_csv(StringIO(content), delim_whitespace=True, escapechar='#', usecols=['D(1,N)', 'FORCE'])
    data.columns = ['d', 'F']
    if not inverse:
        data = data.loc[data['F'] > 0.1]
    else:
        data['F'] = data['F'] * (-1)
        data = data.loc[data['F'] > 0.1]
    data['d'] = data['d'] * 0.1
    data['F'] = data['F'] * 10
    data = data.reset_index(drop=True)
    return data


def smooth_data(data):
    """ Smoothing the input data.

            :param data: data
            :type data: DataFrame
            :return: smoothed data
            :rtype: DataFrame
    """

    range_d = np.linspace(data['d'].min(), data['d'].max(), 1000)
    range_f = []
    d_new = []
    f_new = []

    for i in range(len(range_d) - 1):
        d_min = range_d[i]
        d_max = range_d[i + 1]
        partial_data = data.loc[(data['d'] >= d_min) & (data['d'] <= d_max)]
        range_f.append(list(partial_data['F']))

        if len(range_f[i]) > 0:
            d_new.append(d_min)
            f_new.append(np.mean(range_f[i]))

    d_new, f_new = running_average(d_new, f_new)
    smooth = pd.DataFrame({'d': d_new, 'F': f_new})

    return smooth


def running_average(x, y, window=None):
    """ Finding running average of smoothed x and y values.

    :param x: arguments to be smoothed
    :type x: list or numpy array
    :param y: values to be smoothed
    :type y: list or numpy array
    :param window: window
    :type window: list or numpy array (default None)
    :return: smoothed x values, smoothed y values
    :rtype: two numpy arrays

    """
    if not window:
        window = max(int(len(x) / 100), 8)
    x_smooth = np.convolve(x, np.ones((window,)) / window, mode='valid')
    y_smooth = np.convolve(y, np.ones((window,)) / window, mode='valid')
    return x_smooth, y_smooth


def find_last_range(data, smoothed_data):
    """Finding the approximated range of the last protein state.

    :param parameters:
    :param data: input data
    :type data: DataFrame
    :param smoothed_data: smoothed input data
    :type smoothed_data: DataFrame
    :return: approximated range of the last state
    :rtype: tuple

    """
    extremas = argrelextrema(smoothed_data[smoothed_data['d'] < smoothed_data['d'].max() - 3]['F'].to_numpy(),
                             np.less)[0]
    local_minimum = smoothed_data.loc[extremas[-1], 'd']
    # print(smoothed_data.loc[extremas, 'd'])
    end = smoothed_data['d'].max()
    data_range = data[data['d'].between(local_minimum + 1, end - 1)]
    last_range = (data_range.loc[data_range['F'].idxmin(), 'd'], data_range['d'].max())
    # last_range = (25, 35)
    return last_range


def marko_siggia(d, length, p, k=0):
    """Method that computes the value of extended WLC function in a particular point with three parameters:
    persistence length, force constant and total length of a protein.

    :param d: the extension of a protein
    :type d: float
    :param length: expected total length of a protein
    :type length: float
    :param p: constant (Boltzmann constant times temperature divided by persistence length)
    :type p: float
    :param k: force constant (default: 0)
    :type k: float
    :return: force value in a chosen point of extension
    :rtype: float

    """

    if k == 0:
        return p * (0.25 / ((1 - d / length) ** 2) - 0.25 + d / length)
    else:
        x = d / length
        coefs = [-(k ** 3) - (k ** 2) / p,
                 -(2.25 * (k ** 2)) - 2 * k / p + x * (3 * (k ** 2) + 2 * (k / p)),
                 -(1.5 * k) - 1 / p + x * (4.5 * k + 2 / p) - x ** 2 * (3 * k + 1 / p),
                 1.5 * x - 2.25 * x ** 2 + x ** 3]
        result = np.roots(coefs)
        result = np.real(result[np.isreal(result)])
        result = result[result > 0]

        return max(result)


def wlc(distances, length, p, k=0):
    """

    :param distances: array of extensions
    :type distances: array-like
    :type length: float
    :param p: constant (Boltzmann constant times temperature divided by persistence length)
    :type p: float
    :param k: force constant (default: 0)
    :type k: float
    :return: array of Marko-Siggia function values
    :rtype: numpy array

    """
    return np.array([marko_siggia(d, length, p, k) for d in distances])


def invert_wlc(force, p, k=0):
    """Inverting the Marko-Siggia function in order to find d/length.

    :param force: Force value in given point
    :param p: constant (Boltzmann constant times temperature divided by persistence length)
    :type p: float
    :param k: force constant (default: 0)
    :type k: float
    :return: d/length value
    :rtype: float

    """

    if k == 0:
        coefs = [1, -(2.25 + force / p), (1.5 + 2 * force / p), -force / p]
    else:

        coefs = [1,
                 -(2.25 + force * (3 * k + 1 / p)),
                 (3 * (k ** 2) + 2 * (k / p)) * force ** 2 + ((4.5 * k) + (2 / p)) * force + 1.5,
                 -force * (((k ** 3) + ((k ** 2) / p)) * (force ** 2) + (2.25 * (k ** 2) + 2 * (k / p)) * force + (
                         (1.5 * k) + (1 / p)))]
    # print(k, p, force)
    # print(coefs)
    result = np.roots(coefs)
    result = np.real(result[np.isreal(result)])
    result = result[result > 0]
    if k == 0:
        result = result[result < 1]
    return min(result)


def to_minimize(x, data, last_range, bond_length, residues, exp=False, **kwargs):
    """Creating a function to be minimized.

    :param exp:
    :param x: two-element array: x[0] is the value of p, x[1] is the value of k #### dict!!!
    :type x: list/array
    :param data: input data
    :type data: DataFrame
    :param last_range: last range
    :type last_range: tuple
    :param bond_length: bond length
    :type bond_length: float
    :param residues: number of residues
    :type residues: int
    :return: vector norm
    :rtype: int/float

    """
    print(x)
    fit_data = data[data['d'].between(last_range[0], last_range[1])]
    length = bond_length * (residues - 1)
    fit_f = wlc(fit_data['d'], length, x[0], x[1])
    # pprot pdna kprot kdna ldna
    if exp:
        d_dna = get_d_dna(kwargs['p_dna'], kwargs['l_dna'], kwargs['k_dna'], fit_data['F'].to_numpy())
        fit_data = fit_data.reset_index(drop=True)
        fit_data['d'] = fit_data['d'] - d_dna
        fit_data = fit_data[fit_data['d'] > 0]
        # fit_data = fit_data[fit_data['d'] < x[4]]

    return np.linalg.norm(fit_f - fit_data['F'].to_numpy())


def fit(data, last_range, bond_length, residues, initial_guess, bounds):
    """

    :param data: input data
    :type data: DataFrame
    :param last_range: last range
    :type last_range: tuple
    :param bond_length: bond length
    :type bond_length: float
    :param residues: number of residues
    :type residues: int
    :param initial_guess: initial parameters
    :type initial_guess: array
    :param bounds: bounds of parameters
    :type bounds: tuple of two tuples
    :return: list of two optimal parameters: p and k
    :rtype: list

    """

    opt = minimize(to_minimize, x0=initial_guess, args=(data, last_range, bond_length, residues), method='TNC',
                   bounds=bounds)
    return opt['x']


def find_contour_lengths(data, p, k):
    """Finding contour lengths from input data.

    :param data: input data
    :type data: DataFrame
    :param p: constant (Boltzmann constant times temperature divided by persistence length)
    :type p: float
    :param k: force constant (default: 0)
    :type k: float
    :return: list of contour lengths
    :rtype: list

    """
    x = np.array([invert_wlc(force, p, k) for force in data['F']])
    list_of_contour_lengths = []
    for i in range(len(data)):
        list_of_contour_lengths.append(data['d'][i] / x[i])
    return list_of_contour_lengths


def single_gaussian(x, height, mean, width):
    """

    :param x: argument of gaussian function
    :type x: float or array
    :param height: height of gaussian peak
    :type height: float
    :param mean: mean of gaussian peak
    :type mean: float
    :param width: width of gaussian peak
    :type width: float
    :return: value of gaussian function in point x
    :rtype: float

    """

    return height * np.exp(-(x - mean) ** 2 / (2 * width ** 2))


def multiple_gaussian(x, *args):
    """The function expects 3*states parameters (height1, center1, width1, height2, center2, width2, ...
    :param x: argument of gaussian function
    :type x: float

    :return: value of gaussian function in point x
    :rtype: float

    """

    result = np.zeros(len(x))
    for k in range(0, len(args), 3):
        height, mean, width = args[k], args[k + 1], args[k + 2]
        result += single_gaussian(x, height, mean, width)
    return result


def decompose_histogram(hist_values, significance=0.03, states=None, bandwidth=0.5, **kwargs):
    # sign 0.012 theory
    """Computing number of states, their mean contour lengths, heights, widths.

    :param guess:
    :param bandwidth:
    :param significance:
    :param states:
    :param hist_values: values of the histogram
    :type hist_values: Series
    :return: DataFrame with columns: means, widths, heights
    :rtype: DataFrame

    :Keyword Arguments:
    * *significance* -- (``float``) the precision of finding maximas
    * *states* -- (``int``) predicted number of states (default: none)
    * *bandwidth* -- ?



    """
    x = np.expand_dims(hist_values, 1)
    kde = KernelDensity(bandwidth).fit(x)

    estimator = np.linspace(min(hist_values), max(hist_values), 1001)
    kde_est = np.exp(kde.score_samples(estimator.reshape(-1, 1)))

    if 'guess' in kwargs.keys():
        guesses = kwargs['guess']

    else:

        means = np.array([estimator[_] for _ in argrelextrema(kde_est, np.greater)[0] if kde_est[_] > significance])

        if states:
            missing = max(states - len(means), 0)
            if missing > 0:
                intervals = missing + 1
                beg, end = min(means), max(means)
                additional = np.array([beg * (intervals - i) / intervals + end * i / intervals for i in
                                       range(1, intervals)])
                means = np.append(means, additional)

        heights = np.exp(kde.score_samples(means.reshape(-1, 1)))
        guesses = pd.DataFrame({'heights': heights, 'means': means, 'widths': np.ones(len(means))})

    guesses = guesses.sort_values(by=['means'], ascending=True)
    guesses.index.name = 'state'

    p0 = []
    k = 0
    for ind, row in guesses.iterrows():
        k += 1
        p0 += list(row.values)
    p0 = tuple(p0)

    popt = list()
    while len(popt) < len(p0):
        try:
            popt, pcov = curve_fit(multiple_gaussian, estimator, kde_est, p0=p0)
            popt = list(popt)
        except RuntimeError:
            p0 = p0[:-3]
            print("I reduced the number of states from expected (" + str(int((len(p0) + 3) / 3)) + ") to " + str(
                int(len(p0) / 3)))

    for k in range(0, len(popt), 3):
        if abs(round(popt[k + 1], 3)) == 0.000:
            del popt[k:k + 3]
            break

    popt = tuple(popt)

    parameters = pd.DataFrame({'heights': np.array([round(popt[k], 3) for k in range(0, len(popt), 3)]),
                               'means': np.array([round(popt[k + 1], 3) for k in range(0, len(popt), 3)]),
                               'widths': np.array([abs(round(popt[k + 2], 3)) for k in range(0, len(popt), 3)])})
    parameters = parameters.sort_values(by=['means'])

    # for index, row in parameters.iterrows():
    #     if row['heights'] < 0.012:
    #         parameters.drop([index], inplace=True)
    #         parameters.reset_index(drop=True, inplace=True)

    return parameters


def get_color(index):
    """

    :param index: number of colors index
    :type index: int
    :return: color from mcolors.CSS4_COLORS

    """

    colors = [mcolors.CSS4_COLORS['red'],
              mcolors.CSS4_COLORS['green'],
              mcolors.CSS4_COLORS['blue'],
              mcolors.CSS4_COLORS['yellow'],
              mcolors.CSS4_COLORS['cyan'],
              mcolors.CSS4_COLORS['orange'],
              mcolors.CSS4_COLORS['purple'],
              mcolors.CSS4_COLORS['lime'],
              mcolors.CSS4_COLORS['magenta'],
              mcolors.CSS4_COLORS['olive']]

    return colors[index]


def work(data, begs, ends, inverse=False):
    """Calculating area under the curve; direct method.

    :param data:
    :type data: DataFrame
    :param begs: list of beginnings of states
    :type begs: list
    :param ends: list of ends of states
    :type ends: list
    :return: list of areas between states
    :rtype: list

    """

    work_ = 0
    area = []
    for beg_item, end_item in zip(begs, ends):
        cut_data = data.loc[(data['d'] >= beg_item) & (data['d'] <= end_item)]
        cut_data = cut_data.reset_index(drop=True)
        if cut_data.empty:
            area.append(np.NaN)
            continue
        for i in range(len(cut_data) - 1):
            if inverse:
                F = (cut_data['F'].at[-i] + cut_data['F'].at[-i - 1]) / 2
                dx = cut_data['d'].at[-i] - cut_data['d'].at[-i - 1]
            else:
                F = (cut_data['F'].at[i + 1] + cut_data['F'].at[i]) / 2
                dx = cut_data['d'].at[i + 1] - cut_data['d'].at[i]

            work_ = work_ + F * dx

        work_ = round(work_, 3)
        area.append(work_)
        work_ = 0

    return area


def simpson(data, begs, ends):
    """Calculating area under the curve; numerical Simpson method.

        :param data:
        :type data: DataFrame
        :param begs: list of beginnings of states
        :type begs: list
        :param ends: list of ends of states
        :type ends: list
        :return: list of areas between states
        :rtype: list

        """

    area = []
    for beg_item, end_item in zip(begs, ends):
        cut_data = data.loc[(data['d'] >= beg_item) & (data['d'] <= end_item)]
        cut_data = cut_data.reset_index(drop=True)
        if cut_data.empty:
            area.append(uniform(beg_item, end_item))
            continue
        area.append(round(simps(cut_data['F'].to_numpy(), cut_data['d'].to_numpy()), 3))

    return area


def dhs_feat_cusp(force, x, t0, g):
    if 1 - 0.5 * force.max() * x / g < 0 or t0 < 0:
        return np.array([999 for _ in range(len(force))])
    return np.log(t0) + 1/2 * np.log(1 - force * x / (2 * g)) - g/0.5 * (1 - (1 - force * x / (2 * g))**2)
    # return np.log(t0) + 1 / 2 * np.log(1 - force * x / (2 * g)) - 0.5 * g * (1 - (1 - force * x / (2 * g)) ** 2)
    # return np.log(t0) - x * force / 4.114 - np.log(1 - 0.5 * force * x / g) + ((0.5 * force * x) ** 2) / g


def dhs_feat_linear_cubic(force, x, t0, g):
    return np.log(t0 / (1 - 2 * x / g * force / 3) ** (-1 / 2) * np.exp(-g / 0.5 * (1 - (1 - 2 * x / g * force / 3) **
                                                                                      (3 /
                                                                                       2))))


def dhs_feat_bell(force, x, t0):
    if t0 < 0:
        return np.array([999 for _ in range(len(force))])
    return np.log(t0) - x * force / 0.5
    # 4.114
    # return t0 * np.exp(- x * force)


def integrate_gauss(force, mean, width):
    return 0.5 * (1 - erf((force - mean) / (np.sqrt(width * 2))))


def loading_force(force, p_linker=0.16, l_linker=350, speed=500, k_spring=0.3):
    print(p_linker, l_linker)
    # speed = 0.001
    # speed = 500
    # k = 0.003 ??
    if (p_linker == 0) and (l_linker == 0):
        factor = 1 / k_spring
        return speed / factor

    dna_part = (2 * (l_linker / p_linker) * (1 + force / p_linker)) / (
            3 + 5 * force / p_linker + 8 * (force / p_linker) ** (5 / 2))
    factor = dna_part + 1 / k_spring

    return speed * factor


def get_d_dna(p_dna, l_dna, k_dna, f_space):
    if l_dna > 0:
        column = [l_dna * invert_wlc(f, p_dna, k=k_dna) for f in f_space]
        df = pd.DataFrame({})
        df['d'] = np.array(column)
        return df['d']
    else:
        return np.zeros(len(f_space))


def read_dataframe(input_data, cases=None, columns=None):
    if columns and all([isinstance(c, int) for c in columns]):
        data = input_data.iloc[:, columns]
        data.columns = ['d', 'F']
        data = data.loc[data['F'] > 0.1]
        data = data.reset_index(drop=True)
        return data
    elif columns and not all([isinstance(c, int) for c in columns]):
        return input_data[columns]
    elif cases:
        allowed = [str(_) for _ in cases]
        colnames = [name for name in list(input_data) if name.strip('dF_') in allowed]
        return input_data[colnames]
    else:
        return input_data


def read_excel(input_data, cases, columns):
    data = pd.read_excel(input_data)
    return read_dataframe(data, cases=cases, columns=columns)


def dudko_hummer_szabo(rupture_table, states_rup):
    fig = plt.figure(figsize=(5, 5), dpi=100)
    results = {'x': [], 't0': [], 'g': []}
    for ind, row in rupture_table[['heights', 'means', 'widths']][:-1].iterrows():
        f_space = np.linspace(min(list(states_rup.values())[ind]), row['means'], 1000)
        height, mean, width = tuple(row.to_numpy())
        # sym: speed = 0.0001, k = 0.3 reszta 0

        dhs_data = pd.DataFrame({'forces': f_space,
                                 # 'force_load': loading_force(f_space),
                                 'force_load': loading_force(f_space, p_linker=0, l_linker=0, speed=0.0001,
                                                             k_spring=0.3), # symulacje
                                 'probability': norm.pdf(f_space, mean, width),
                                 'nominator': integrate_gauss(f_space, mean, width)})

        dhs_data['denominator'] = dhs_data['probability'] * dhs_data['force_load']
        # dhs_data = dhs_data[dhs_data['denominator'] > 0.1]
        dhs_data['lifetime'] = dhs_data['nominator'] / dhs_data['denominator']
        # print(dhs_data)
#         coefficients = {}
#         # sym init lifetime 10^15 init x zawsze 1
#         init_lifetime = 5
#         init_x = 1

#         # v = 1
#         p0 = (init_x, init_lifetime)
        # popt_b, pcov_b = curve_fit(dhs_feat_bell, dhs_data['forces'], np.log(dhs_data['lifetime']), p0=p0)
        # print(popt_b, pcov_b)
        # plt.plot(dhs_data['forces'], np.log(dhs_data['lifetime']))
        # plt.show()
        # coefficients['bell'] = {'x': popt_b[0], 't0': popt_b[1], 'g': np.NaN}  # 'covariance': pcov}

        # v = 1/2
        # p0 = (coefficients['bell']['x'], coefficients['bell']['t0'])

        #tm15 exp

#         p0 = (1, 50, 30)
#         bounds = ([0.1, 0.1, 10], [4, 1000, 250])
        # lc exp
        # p0 = (0.75, 50, 25)
        # bounds = ([0.1, 0.5, 10], [4, 1000, 200])
        # cusp exp
        # p0=
        # linear cubic theory
        # p0 = (0.5, 10.05 ** 13, 30)
        # bounds = ([0.1, 10 ** 11, 5], [1.5, 3*10 ** 15, 100])
        # cusp theory
        # p0 = [(1.5, 2.8 * 10 ** 15, 30), (0.62, 1.5 * 10 ** 14, 63), (0.4, 5.3 * 10 ** 14, 37),
        #       (0.4, 1.4 * 10 ** 15, 49)]
        # bounds = [([1.3, 10 ** 15, 25], [1.6, 3 * 10 ** 15, 35]), ([0.5, 10 ** 14, 60], [0.7, 1.8 * 10 ** 14, 70]),
        #           ([0.2, 5 * 10 ** 14, 30], [0.5, 6 * 10 ** 14, 40]),
        #           ([0.2, 1 * 10 ** 15, 40], [0.5, 2 * 10 ** 15, 55])]
        # p0 = (1, 10 ** 13, 50)
        # bounds = ([0.2, 10 ** 11, 30], [2, 1.5*10 ** 15, 70])
        # bounds = ([0.2, 10 ** 13, 20], [5, 10 ** 16, 80])
#         print("FORCES")

#         with pd.option_context('display.max_rows', None, 'display.max_columns',
#                                None):  # more options can be specified also
#             print(dhs_data['forces'])

#         with pd.option_context('display.max_rows', None, 'display.max_columns',
#                                None):  # more options can be specified also
#             print(dhs_data['lifetime'])

    #     try:
    #         popt, pcov = curve_fit(dhs_feat_linear_cubic, dhs_data['forces'], np.log(dhs_data['lifetime']), p0=p0,
    #                                bounds=bounds)
    #         print(popt, pcov)
    #         # popt, pcov = curve_fit(dhs_feat_cusp, dhs_data['forces'], np.log(dhs_data['lifetime']), p0=p0,
    #         #                        bounds=bounds)
    #         coefficients['lc'] = {'x': popt[0], 't0': popt[1], 'g': popt[2]}  # , 'covariance': pcov}
    #         # print(coefficients['linear_cubic'])
    #         print('xd')
    #
    #     except RuntimeError:
    #         result = None
    #
    #
    #         # plt.plot(f_space, dhs_feat_cusp(f_space, coefficients['cusp']['x'], coefficients['cusp']['t0'],
    #         #                                 coefficients['cusp']['g']), color='black')
    #     results['x'].append(coefficients['lc']['x'])
    #     results['t0'].append(coefficients['lc']['t0'])
    #     results['g'].append(coefficients['lc']['g'])
    #
    #     #label = 'F = ' + str(round(row['means'], 3)) + ' pN'
    #     label = 'state ' + str(ind+1)
    #
    #     plt.plot(dhs_data['forces'], np.log(dhs_data['lifetime']), color=get_color(ind), label=label)
    #     # plt.plot(f_space, dhs_feat_linear_cubic(f_space, popt_b[0], popt_b[1],
    #     #                                 ), color='black', ls='-.')
    #     plt.plot(f_space, dhs_feat_linear_cubic(f_space, popt[0], popt[1], popt[2]), color='black', ls='-.')
    #
    # plt.title('Dudko-Hummer-Szabo lifetimes')
    # # plt.xlabel(r'Rupture force [$\frac{\epsilon}{nm}$]')
    # plt.xlabel('Rupture force [pN]')
    # plt.ylabel('log(state lifetime)')
    # plt.legend(fontsize='small')
    # plt.savefig("Images/dudko.png")
    #
    # dhs = pd.DataFrame.from_dict(results)

    return None # dhs


def load_data_new(path):
    data = pd.read_csv(path, names=["d", "fold", "F"], delim_whitespace=True)
    print(data)
    data = data.loc[data['F'] > 0]
    data['d'] = data['d']
    data = data.reset_index(drop=True)
    # data['d'] = data['d'] - min(data['d'])
    # print(data)
    return data
