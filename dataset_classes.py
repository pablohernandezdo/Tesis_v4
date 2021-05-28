import re
import os
import h5py
import segyio
import numpy as np
import scipy.io as sio
from numpy.random import default_rng

from obspy.io.seg2 import seg2
from obspy.io.segy.core import _read_segy
from obspy.signal.trigger import classic_sta_lta

from scipy import signal
from scipy.signal import butter, lfilter, detrend


class Dsets:
    def preprocess(self, traces, fs):

        new_traces = []
        N_new = int(len(traces[0]) * 100 / fs)

        for trace in traces:

            if np.sum(np.abs(trace)) == 0:
                continue

            if fs / 2 < 50:
                # Filter 50 Hz
                trace = self.butter_lowpass_filter(trace, 50, fs)

            # Detrending
            trace = detrend(trace)

            # Media cero
            trace = trace - np.mean(trace)

            # Remuestrear a 100 hz
            if fs != 100:
                trace = signal.resample(trace, N_new)

            new_traces.append(trace)

        return np.asarray(new_traces)

    @staticmethod
    def save_dataset(traces, savepath, name):
        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        np.save(f'{savepath}/{name}', traces)

    @staticmethod
    def normalize(traces):
        norm_traces = []

        for trace in traces:
            if np.amax(np.abs(trace)):
                trace /= np.amax(np.abs(trace))

            norm_traces.append(trace)

        return np.asarray(norm_traces)

    @staticmethod
    def butter_lowpass_filter(dat, highcut, fs, order=5):
        nyq = 0.5 * fs
        high = highcut / nyq
        b, a = butter(order, high, output='ba')
        y = lfilter(b, a, dat)
        return y

    @staticmethod
    def read_segy(dataset_path):
        with segyio.open(dataset_path, ignore_geometry=True) as segy:
            # Memory map, faster
            segy.mmap()

            # Traces and sampling frequency
            traces = segyio.tools.collect(segy.trace[:])
            fs = segy.header[0][117]

        return traces, fs


class DatasetFrancia(Dsets):
    def __init__(self, dataset_path, savepath, unproc_savepath):
        super(DatasetFrancia, self).__init__()

        self.dataset_path = dataset_path
        self.savepath = savepath
        self.unproc_savepath = unproc_savepath

        print(f"Reading dataset from path: {self.dataset_path}")
        self.traces = sio.loadmat(self.dataset_path)["StrainFilt"]
        self.fs = 100

        print("Saving npy unprocessed dataset")
        if not os.path.exists(f'{self.unproc_savepath}/Francia.npy'):
            self.save_dataset(self.traces, self.unproc_savepath, 'Francia')

        print("Preprocessing dataset")
        self.traces = self.preprocess(self.traces, self.fs)

        print("Normalizing dataset")
        self.traces = self.normalize(self.traces)

        print(f"Saving npy format dataset in {self.savepath}")
        if not os.path.exists(f'{self.savepath}/Francia.npy'):
            self.save_dataset(self.traces, self.savepath, 'Francia')

    def prune_traces(self):
        pass


class DatasetNevada(Dsets):
    def __init__(self, dataset_path, savepath, unproc_savepath):
        super(DatasetNevada, self).__init__()

        self.dataset_path = dataset_path
        self.savepath = savepath
        self.unproc_savepath = unproc_savepath

        print(f"Reading dataset from path: {self.dataset_path}")
        self.traces, self.fs = self.read_segy(self.dataset_path)

        print("Saving npy unprocessed dataset")
        if not os.path.exists(f'{self.unproc_savepath}/Nevada.npy'):
            self.save_dataset(self.traces, self.unproc_savepath, 'Nevada')

        print("Preprocessing dataset")
        self.traces = self.preprocess(self.traces, self.fs)

        print("Padding traces")
        self.traces = self.padd()

        print("Normalizing dataset")
        self.traces = self.normalize(self.traces)

        print(f"Saving npy format dataset in {self.savepath}")
        if not os.path.exists(f'{self.savepath}/Nevada.npy'):
            self.save_dataset(self.traces, self.savepath, 'Nevada')

    def padd(self):

        rng = default_rng()
        n_padd = 6000 - self.traces.shape[1]

        padd_traces = []

        for trace in self.traces:
            # 30 ventanas de 100 muestras
            windows = trace.reshape(30, 100)

            # calcular la varianza de ventanas
            stds = np.std(windows, axis=1)

            # generar ruido y padd
            ns = rng.normal(0, np.amin(stds) / np.sqrt(2), n_padd)
            # trace = np.hstack([trace, ns])
            trace = np.hstack([ns, trace])
            padd_traces.append(trace)

        return np.asarray(padd_traces)


class DatasetBelgica(Dsets):
    def __init__(self, dataset_path, savepath):
        super(DatasetBelgica, self).__init__()

        self.dataset_path = dataset_path
        self.savepath = savepath
        self.fs = 10
        self.n_traces = 7000

        # Dataset de ruido, no es necesario preprocesar
        print(f"Reading dataset from path: {self.dataset_path}")
        self.traces = sio.loadmat(self.dataset_path)["Data_2D"]

        self.noise_traces = np.empty((0, 6000))

        # ventanas en tiempo
        sta_t = 3
        lta_t = 25

        # ventanas en muestras
        sta_n = sta_t * self.fs * 10
        lta_n = lta_t * self.fs * 10

        copied = 0

        for trace in self.traces:
            trace = trace.reshape(-1, 6000)

            for tr in trace:
                tr = detrend(tr)
                tr = tr - np.mean(tr)
                tr /= np.amax(tr)

                cft = classic_sta_lta(tr, sta_n, lta_n)

                if np.amax(cft) < 2:
                    self.noise_traces = np.vstack([self.noise_traces, tr])
                    copied += 1

                if not (copied % 100):
                    print(f"copied: {copied}")

                if copied == self.n_traces:
                    break

            if copied == self.n_traces:
                break

        print(f"Saving npy format dataset in {self.savepath}")
        if not os.path.exists(f'{self.savepath}/Belgica.npy'):
            self.save_dataset(self.noise_traces, self.savepath, 'Belgica')


class DatasetReykjanes(Dsets):
    def __init__(self, dataset_path, savepath, unproc_savepath):
        super(DatasetReykjanes, self).__init__()

        self.dataset_path = dataset_path
        self.savepath = savepath
        self.unproc_savepath = unproc_savepath
        self.fs = 200
        self.n_traces = 2551

        print(f"Reading dataset from path: {self.dataset_path}")
        self.header, self.traces = self.read_ascii()

        print("Saving npy unprocessed dataset")
        if not os.path.exists(f'{self.unproc_savepath}/Reykjanes.npy'):
            self.save_dataset(self.traces, self.unproc_savepath, 'Reykjanes')

        print("Preprocessing dataset")
        self.traces = self.preprocess(self.traces, self.fs)

        print("Padding traces")
        self.traces = self.padd()

        print("Normalizing dataset")
        self.traces = self.normalize(self.traces)

        print(f"Saving npy format dataset in {self.savepath}")
        if not os.path.exists(f'{self.savepath}/Reykajnes.npy'):
            self.save_dataset(self.traces, self.savepath, 'Reykjanes')

    def padd(self):
        rng = default_rng()
        n_padd = 6000 - self.traces.shape[1]

        padd_traces = []

        for trace in self.traces:
            # 14 ventanas de 50 muestras
            windows = trace.reshape(14, 50)

            # calcular la varianza de ventanas
            stds = np.std(windows, axis=1)

            # generar ruido y padd
            ns = rng.normal(0, np.amin(stds) / 4, n_padd)
            trace = np.hstack([trace, ns])
            padd_traces.append(trace)

        return np.asarray(padd_traces)

    def read_ascii(self):
        # Preallocate
        traces = np.empty((1, self.n_traces))

        with open(self.dataset_path, 'r') as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    header = line.strip()

                else:
                    row = np.asarray(list(
                        map(float, re.sub(' +', ' ', line).strip().split(' '))))
                    traces = np.concatenate((traces, np.expand_dims(row, 0)))

        # Delete preallocate empty row and transpose
        traces = traces[1:]
        traces = traces.transpose()

        return header, traces


class DatasetCalifornia(Dsets):
    def __init__(self, dataset_paths, savepath, unproc_savepath):
        super(DatasetCalifornia, self).__init__()

        if len(dataset_paths) != 4:
            print("Se necesitan 4 archivos!")

        else:
            self.savepath = savepath
            self.unproc_savepath = unproc_savepath
            self.dataset_paths = dataset_paths
            self.fs = 1000
            self.d1, self.d2, self.d3, self.d4 = self.dataset_paths

            print(f"Reading datasets from path {os.path.dirname(self.d1)}")
            self.traces_d1 = sio.loadmat(self.d1)['singdecmatrix'].T
            self.traces_d2 = sio.loadmat(self.d2)['singdecmatrix'].T
            self.traces_d3 = sio.loadmat(self.d3)['singdecmatrix'].T
            self.traces_d4 = sio.loadmat(self.d4)['singdecmatrix'].T

            print("Saving npy unprocessed datasets")
            if not os.path.exists(f'{self.unproc_savepath}/California1.npy'):
                self.save_dataset(self.traces_d1, self.unproc_savepath,
                                  'California1')

            if not os.path.exists(f'{self.unproc_savepath}/California2.npy'):
                self.save_dataset(self.traces_d2, self.unproc_savepath,
                                  'California2')

            if not os.path.exists(f'{self.unproc_savepath}/California3.npy'):
                self.save_dataset(self.traces_d3, self.unproc_savepath,
                                  'California3')

            if not os.path.exists(f'{self.unproc_savepath}/California4.npy'):
                self.save_dataset(self.traces_d4, self.unproc_savepath,
                                  'California4')

            print(f"Preprocessing datasets")
            self.traces_d1 = self.preprocess(self.traces_d1, self.fs)
            self.traces_d2 = self.preprocess(self.traces_d2, self.fs)
            self.traces_d3 = self.preprocess(self.traces_d3, self.fs)
            self.traces_d4 = self.preprocess(self.traces_d4, self.fs)

            print(f"Reordering datasets traces")
            self.traces_d1 = self.trim(self.traces_d1)
            self.traces_d2 = self.trim(self.traces_d2)
            self.traces_d3 = self.trim(self.traces_d3)
            self.traces_d4 = self.trim(self.traces_d4)

            print("Normalizing datasets")
            self.traces_d1 = self.normalize(self.traces_d1)
            self.traces_d2 = self.normalize(self.traces_d2)
            self.traces_d3 = self.normalize(self.traces_d3)
            self.traces_d4 = self.normalize(self.traces_d4)

            print("Stacking datasets")
            self.traces = np.vstack([self.traces_d1,
                                     self.traces_d2,
                                     self.traces_d3,
                                     self.traces_d4])

            print(f"Saving npy format individual datasets in {self.savepath}")
            if not os.path.exists(f'{self.savepath}/California1.npy'):
                self.save_dataset(self.traces_d1, self.savepath, 'California1')

            if not os.path.exists(f'{self.savepath}/California2.npy'):
                self.save_dataset(self.traces_d2, self.savepath, 'California2')

            if not os.path.exists(f'{self.savepath}/California3.npy'):
                self.save_dataset(self.traces_d3, self.savepath, 'California3')

            if not os.path.exists(f'{self.savepath}/California4.npy'):
                self.save_dataset(self.traces_d4, self.savepath, 'California4')

            print(f"Saving npy format dataset in {self.savepath}")
            if not os.path.exists(f'{self.savepath}/California.npy'):
                self.save_dataset(self.traces, self.savepath, 'California')

    @staticmethod
    def trim(traces):
        traces = traces[:, :traces.shape[1] // 6000 * 6000]
        traces = traces.reshape((-1, 6000))
        return traces


class DatasetHydraulic(Dsets):
    def __init__(self, dataset_path, savepath, unproc_savepath):
        super(DatasetHydraulic, self).__init__()

        self.savepath = savepath
        self.unproc_savepath = unproc_savepath
        self.dataset_path = dataset_path

        print(f"Reading dataset from path: {self.dataset_path}")
        self.fs, self.traces = self.read_file()

        print("Saving npy unprocessed dataset")
        if not os.path.exists(f'{self.unproc_savepath}/Hydraulic.npy'):
            self.save_dataset(self.traces, self.unproc_savepath, 'Hydraulic')

        print("Preprocessing dataset")
        self.traces = self.preprocess(self.traces, self.fs)

        print(f"Reordering datasets traces")
        self.traces = self.trim()

        print("Normalizing dataset")
        self.traces = self.normalize(self.traces)

        print(f"Saving npy format dataset in {self.savepath}")
        if not os.path.exists(f'{self.savepath}/Hydraulic.npy'):
            self.save_dataset(self.traces, self.savepath, 'Hydraulic')

    def trim(self):
        # Repetir última muestra para que sean 120_000
        traces = np.hstack([self.traces, self.traces[:, -1].reshape(-1, 1)])
        traces = traces.reshape(-1, 6000)
        return traces

    def read_file(self):
        with h5py.File(self.dataset_path, 'r') as f:
            traces = f['data'][()]
            fs = f['fs_f'][()].item()
        return fs, traces


class DatasetVibroseis(Dsets):
    def __init__(self, dataset_paths, savepath, unproc_savepath):
        super(DatasetVibroseis, self).__init__()

        if len(dataset_paths) != 4:
            print("Se necesitan 4 archivos!")

        else:
            self.savepath = savepath
            self.unproc_savepath = unproc_savepath
            self.dataset_paths = dataset_paths
            self.fs = 1000
            self.d1, self.d2, self.d3, self.d4 = self.dataset_paths

            print(f"Reading datasets from path {os.path.dirname(self.d1)}")
            self.traces_d1, self.fs = self.read_segy(self.d1)
            self.traces_d2, _ = self.read_segy(self.d2)
            self.traces_d3, _ = self.read_segy(self.d3)
            self.traces_d4, _ = self.read_segy(self.d4)

            print("Saving npy unprocessed datasets")
            if not os.path.exists(f'{self.unproc_savepath}/Vibroseis1.npy'):
                self.save_dataset(self.traces_d1, self.unproc_savepath,
                                  'Vibroseis1')

            if not os.path.exists(f'{self.unproc_savepath}/Vibroseis2.npy'):
                self.save_dataset(self.traces_d2, self.unproc_savepath,
                                  'Vibroseis2')

            if not os.path.exists(f'{self.unproc_savepath}/Vibroseis3.npy'):
                self.save_dataset(self.traces_d3, self.unproc_savepath,
                                  'Vibroseis3')

            if not os.path.exists(f'{self.unproc_savepath}/Vibroseis4.npy'):
                self.save_dataset(self.traces_d4, self.unproc_savepath,
                                  'Vibroseis4')

            print(f"Preprocessing datasets")
            self.traces_d1 = self.preprocess(self.traces_d1, self.fs)
            self.traces_d2 = self.preprocess(self.traces_d2, self.fs)
            self.traces_d3 = self.preprocess(self.traces_d3, self.fs)
            self.traces_d4 = self.preprocess(self.traces_d4, self.fs)

            print(f"Padding datasets traces")
            self.traces_d1 = self.padd(self.traces_d1)
            self.traces_d2 = self.padd(self.traces_d2)
            self.traces_d3 = self.padd(self.traces_d3)
            self.traces_d4 = self.padd(self.traces_d4)

            print("Normalizing datasets")
            self.traces_d1 = self.normalize(self.traces_d1)
            self.traces_d2 = self.normalize(self.traces_d2)
            self.traces_d3 = self.normalize(self.traces_d3)
            self.traces_d4 = self.normalize(self.traces_d4)

            print("Stacking datasets")
            self.traces = np.vstack([self.traces_d1,
                                     self.traces_d2,
                                     self.traces_d3,
                                     self.traces_d4])

            print(f"Saving npy format individual datasets in {self.savepath}")
            if not os.path.exists(f'{self.savepath}/Vibroseis1.npy'):
                self.save_dataset(self.traces_d1, self.savepath, 'Vibroseis1')

            if not os.path.exists(f'{self.savepath}/Vibroseis2.npy'):
                self.save_dataset(self.traces_d2, self.savepath, 'Vibroseis2')

            if not os.path.exists(f'{self.savepath}/Vibroseis3.npy'):
                self.save_dataset(self.traces_d3, self.savepath, 'Vibroseis3')

            if not os.path.exists(f'{self.savepath}/Vibroseis4.npy'):
                self.save_dataset(self.traces_d4, self.savepath, 'Vibroseis4')

            print(f"Saving npy format dataset in {self.savepath}")
            if not os.path.exists(f'{self.savepath}/Vibroseis.npy'):
                self.save_dataset(self.traces, self.savepath, 'Vibroseis')

    @staticmethod
    def padd(traces):
        rng = default_rng()
        n_padd = 6000 - traces.shape[1]

        padd_traces = []

        for trace in traces:
            # 30 ventanas de 100 muestras
            windows = trace.reshape(30, 100)

            # calcular la varianza de ventanas
            stds = np.std(windows, axis=1)

            # generar ruido y padd
            ns = rng.normal(0, np.amin(stds) / 4, n_padd)
            trace = np.hstack([trace, ns])
            padd_traces.append(trace)

        return np.asarray(padd_traces)


class DatasetShaker(Dsets):
    def __init__(self, dataset_path, savepath, unproc_savepath):
        super(DatasetShaker, self).__init__()

        self.savepath = savepath
        self.unproc_savepath = unproc_savepath
        self.dataset_path = dataset_path

        print(f"Reading dataset from path: {self.dataset_path}")
        with segyio.open(self.dataset_path, ignore_geometry=True) as segy:
            segy.mmap()
            self.traces = segyio.tools.collect(segy.trace[:])

        self.fs = 200

        print("Saving npy unprocessed dataset")
        if not os.path.exists(f'{self.unproc_savepath}/Shaker.npy'):
            self.save_dataset(self.traces, self.unproc_savepath, 'Shaker')

        print("Preprocessing dataset")
        self.traces = self.preprocess(self.traces, self.fs)

        print(f"Reordering datasets traces")
        self.traces = self.trim()

        print("Normalizing dataset")
        self.traces = self.normalize(self.traces)

        print(f"Saving npy format dataset in {self.savepath}")
        if not os.path.exists(f'{self.savepath}/Shaker.npy'):
            self.save_dataset(self.traces, self.savepath, 'Shaker')

    def trim(self):
        traces = self.traces[:, :self.traces.shape[1] // 6000 * 6000]
        traces = traces.reshape(-1, 6000)
        return traces


class DatasetCoompana(Dsets):
    def __init__(self, dataset_path, savepath):
        super(DatasetCoompana, self).__init__()

        self.savepath = savepath
        self.dataset_path = dataset_path
        self.fs = 4000

        print(f"Reading dataset from path: {self.dataset_path}")

        seg2reader = seg2.SEG2()

        traces_6k = []
        traces_8k = []

        # Every data folder
        for fold in os.listdir(self.dataset_path):

            if os.path.isdir(f"{self.dataset_path}/{fold}"):

                # Read every file
                for datafile in os.listdir(f"{self.dataset_path}/{fold}"):

                    data = seg2reader.read_file(
                        f"{self.dataset_path}/{fold}/{datafile}")

                    # To ndarray
                    for wave in data:
                        # read wave data
                        trace = wave.data

                        # Hay trazas de 6000 y 8000 muestras
                        if trace.size == 6000:
                            traces_6k.append(trace)

                        else:
                            traces_8k.append(trace)

        self.traces_6k = np.asarray(traces_6k)
        self.traces_8k = np.asarray(traces_8k)

        print("Padding 6k sample traces to 8k and stacking")
        self.padd_6k()
        self.traces = np.vstack([self.traces_6k, self.traces_8k])

        print("Preprocessing dataset")
        self.traces = self.preprocess(self.traces, self.fs)

        print("Padding dataset traces")
        self.traces = self.padd()

        print("Normalizing dataset")
        # self.traces = self.normalize(self.traces)
        self.traces = self.normalize(self.traces)

        print(f"Saving npy format dataset in {self.savepath}")
        if not os.path.exists(f'{self.savepath}/Coompana.npy'):
            self.save_dataset(self.traces[:18000, :], self.savepath, 'Coompana')

    def padd(self):

        rng = default_rng()
        n_padd = 6000 - self.traces.shape[1]

        padd_traces = []

        for trace in self.traces:
            # 20 ventanas de 10 muestras
            windows = trace.reshape(20, 10)

            # calcular la varianza de ventanas
            stds = np.std(windows, axis=1)

            # generar ruido y padd
            ns = rng.normal(0, np.amin(stds) / 4, n_padd)
            trace = np.hstack([trace, ns])
            padd_traces.append(trace)

        return np.asarray(padd_traces)

    def padd_6k(self):

        rng = default_rng()
        padded_6k = []

        for trace in self.traces_6k:
            # 60 ventanas de 100 muestras
            windows = trace.reshape(60, 100)

            # calcular la varianza de ventanas
            stds = np.std(windows, axis=1)

            # generar ruido y padd
            ns = rng.normal(0, np.amin(stds) / 4, 2000)
            trace = np.hstack([trace, ns])
            padded_6k.append(trace)

        self.traces_6k = np.asarray(padded_6k)


class DatasetLesser(Dsets):
    def __init__(self, dataset_path, savepath):
        super(DatasetLesser, self).__init__()

        self.savepath = savepath
        self.dataset_path = dataset_path
        self.fs = 250

        traces = []

        print(f"Reading dataset from path: {self.dataset_path}")

        # For every file in the dataset folder
        for dataset in os.listdir(self.dataset_path):

            if dataset.split('.')[-1] == 'segy':

                # Read dataset
                data = _read_segy(f'{self.dataset_path}/{dataset}')

                # For every trace in the dataset
                for wave in data:
                    # To ndarray
                    trace = wave.data

                    # Append to traces list
                    traces.append(trace)

        self.traces = np.asarray(traces)

        print("Preprocessing dataset")
        self.traces = self.preprocess(self.traces, self.fs)

        print("Normalizing dataset")
        # self.traces = self.normalize(self.traces)
        self.traces = self.normalize(self.traces)

        print(f"Saving npy format dataset in {self.savepath}")
        if not os.path.exists(f'{self.savepath}/Lesser.npy'):
            self.save_dataset(self.traces[:18000, :], self.savepath, 'Lesser')


class DatasetNCAirgun(Dsets):
    def __init__(self, dataset_path, savepath):
        super(DatasetNCAirgun, self).__init__()

        self.savepath = savepath
        self.dataset_path = dataset_path
        self.fs = 100

        print(f"Reading dataset from path: {self.dataset_path}")
        data = _read_segy(self.dataset_path)

        traces = []

        for wave in data:
            traces.append(wave.data)

        self.traces = np.array(traces)

        print("Preprocessing dataset")
        self.traces = self.preprocess(self.traces, self.fs)

        print("Normalizing dataset")
        # self.traces = self.normalize(self.traces)
        self.traces = self.normalize(self.traces)

        print(f"Saving npy format dataset in {self.savepath}")
        if not os.path.exists(f'{self.savepath}/NCAirgun.npy'):
            self.save_dataset(self.traces[:18000, :], self.savepath, 'NCAirgun')

