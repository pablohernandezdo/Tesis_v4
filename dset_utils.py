import os
import tqdm
import h5py
import json
import numpy as np
from numpy.random import default_rng


class H5Builder:
    """
    Tomar datasets en formato npy y construir un dataset HDF5, guardarlo
    """

    def __init__(self, config_file):

        # Load dataset config
        with open(config_file, "r") as config:
            self.cfg = json.load(config)

        if not os.path.exists(os.path.dirname(self.cfg["OutputFile"])):
            os.makedirs(os.path.dirname(self.cfg["OutputFile"]), exist_ok=True)

        # Create hdf5 dataset file
        self.h5 = h5py.File(self.cfg["OutputFile"], "w")

        # Create seismic and non-seismic groups
        g_earthquake = self.h5.create_group("earthquake/local")
        g_non_earthquake = self.h5.create_group("non_earthquake/noise")

        # For every dataset in config file
        for dset in self.cfg["Datasets"]:

            print(f"Loading {dset} dataset")
            dset_params = self.cfg["Datasets"][dset]

            # read npy datasets
            if dset_params["format"] == "npy":

                traces = np.load(dset_params["path"])
                if dset_params["type"] == "seismic":
                    for i, tr in enumerate(traces):
                        # tr = np.expand_dims(tr, 1)
                        # tr = np.hstack([tr] * 3).astype("float32")
                        g_earthquake.create_dataset(f"{dset}-{i}",
                                                    data=tr.astype(np.float32))

                elif dset_params["type"] == "nonseismic":
                    for i, tr in enumerate(traces):
                        # tr = np.expand_dims(tr, 1)
                        # tr = np.hstack([tr] * 3).astype("float32")
                        g_non_earthquake.create_dataset(f"{dset}-{i}",
                                                        data=tr.astype(np.float32))

                else:
                    print(f"Bad dataset type! Dataset: {dset}")

            # read h5py datasets
            elif dset_params["format"] == "hdf5":
                with h5py.File(dset_params["path"], "r") as h5_dset:

                    g_source_earthquake = h5_dset["earthquake/local"]
                    g_source_non_earthquake = h5_dset["non_earthquake/noise"]

                    # Copy seismic traces
                    for arr in g_source_earthquake:
                        data = g_source_earthquake[arr]
                        g_earthquake.copy(data, arr)

                    # Copy non seismic traces
                    for arr in g_source_non_earthquake:
                        data = g_source_non_earthquake[arr]
                        g_non_earthquake.copy(data, arr)

            else:
                print(f"Bad dataset format! Dataset: {dset}")

        self.h5.close()


class H5Splitter:
    """
    Tomar un dataset hdf5 y dividirlo en conjuntos de entrenamiento, validacion
    y prueba según una proporción especifica

    Deberia hacer un shuffle de los datos primero, despues armar los datasets
    que haga falta
    """

    def __init__(self, dataset_path, ratios):

        self.dataset_path = dataset_path
        self.dataset_name = self.dataset_path.split("/")[-1].split(".")[0]

        assert len(ratios) == 3, "Ratios must have 3 values!"

        self.train_ratio = ratios[0]
        self.val_ratio = ratios[1]
        self.test_ratio = ratios[2]

        # Cargar dataset de origen
        self.source_h5 = h5py.File(self.dataset_path, "r")
        self.source_seismic_group = self.source_h5["earthquake/local"]
        self.source_nonseismic_group = self.source_h5["non_earthquake/noise"]

        # Numero de trazas sismicas y no sismicas
        self.source_seismic_len = len(self.source_seismic_group)
        self.source_nonseismic_len = len(self.source_nonseismic_group)

        # Calcular el numero de trazas para cada uno
        self.train, self.val, self.test = self.get_traces_division()

        dsets_dir = f'{os.path.dirname(self.dataset_path)}'

        train_name = f"{self.dataset_name}_{self.train_ratio}_train.hdf5"
        val_name = f"{self.dataset_name}_{self.val_ratio}_val.hdf5"
        test_name = f"{self.dataset_name}_{self.test_ratio}_test.hdf5"

        with h5py.File(dsets_dir + '/' + train_name, "w") as train_h5, \
                h5py.File(dsets_dir + '/' + val_name, "w") as val_h5, \
                h5py.File(dsets_dir + '/' + test_name, "w") as test_h5:

            # Create groups
            grp_train_seismic = train_h5.create_group("earthquake/local")
            grp_train_nonseismic = train_h5.create_group("non_earthquake/noise")

            grp_val_seismic = val_h5.create_group("earthquake/local")
            grp_val_nonseismic = val_h5.create_group("non_earthquake/noise")

            grp_test_seismic = test_h5.create_group("earthquake/local")
            grp_test_nonseismic = test_h5.create_group("non_earthquake/noise")

            # Recorrer trazas sismicas del dataset
            for i, dset in enumerate(self.source_seismic_group):

                if i in self.train["seismic"]:
                    grp_train_seismic.copy(self.source_seismic_group[dset],
                                           dset)

                elif i in self.val["seismic"]:
                    grp_val_seismic.copy(self.source_seismic_group[dset], dset)

                elif i in self.test["seismic"]:
                    grp_test_seismic.copy(self.source_seismic_group[dset], dset)

                else:
                    print("Index not found!")

            # Recorrer trazas no sismicas del dataet
            for i, dset in enumerate(self.source_nonseismic_group):

                if i in self.train["nonseismic"]:
                    grp_train_nonseismic.copy(
                        self.source_nonseismic_group[dset],
                        dset)

                elif i in self.val["nonseismic"]:
                    grp_val_nonseismic.copy(self.source_nonseismic_group[dset],
                                            dset)

                elif i in self.test["nonseismic"]:
                    grp_test_nonseismic.copy(self.source_nonseismic_group[dset],
                                             dset)
                else:
                    print("Index not found!")

        self.source_h5.close()

    def get_traces_division(self):

        # Seismic division
        seis_divs = self.source_seismic_len * np.array([self.train_ratio,
                                                        self.val_ratio,
                                                        self.test_ratio])

        seis_divs = np.floor(seis_divs).astype(np.int32)

        # Non-seismic division
        nonseis_divs = self.source_nonseismic_len * np.array([self.train_ratio,
                                                              self.val_ratio,
                                                              self.test_ratio])

        nonseis_divs = np.floor(nonseis_divs).astype(np.int32)

        # Elegir los indices de las trazas sismicas a copiar
        rng = default_rng()

        # Se crean una lista con todos los indices, se eligen los de train
        # y se quitan de a lista
        source_ids = list(range(self.source_seismic_len))
        seismic_train_ids = rng.choice(source_ids, seis_divs[0], replace=False)
        source_ids = list(set(source_ids)-set(seismic_train_ids))

        # Se eligen los ids de val y se quitan de la lista
        seismic_val_ids = rng.choice(source_ids, seis_divs[1], replace=False)
        source_ids = list(set(source_ids) - set(seismic_val_ids))

        # Los ids que quedan son para test
        seismic_test_ids = source_ids

        assert_msg = f"seismic_train_ids: {len(seismic_train_ids)}\n" \
                     f"seismic_val_ids: {len(seismic_val_ids)}\n" \
                     f"seismic_test_ids: {len(seismic_test_ids)}\n" \
                     f"source_ids: {self.source_seismic_len}"

        assert len(seismic_train_ids) + len(seismic_val_ids) + len(
            seismic_test_ids) == self.source_seismic_len, assert_msg

        # Se crean una lista con todos los indices, se eligen los de train
        # y se quitan de a lista
        source_ids = list(range(self.source_nonseismic_len))
        nonseismic_train_ids = rng.choice(source_ids, nonseis_divs[0],
                                          replace=False)
        source_ids = list(set(source_ids) - set(nonseismic_train_ids))

        # Se eligen los ids de val y se quitan de la lista
        nonseismic_val_ids = rng.choice(source_ids, nonseis_divs[1],
                                        replace=False)
        source_ids = list(set(source_ids) - set(nonseismic_val_ids))

        # Los ids que quedan son para test
        nonseismic_test_ids = source_ids

        assert_msg = f"nonseismic_train_ids: {len(nonseismic_train_ids)}\n" \
                     f"nonseismic_val_ids: {len(nonseismic_val_ids)}\n" \
                     f"nonseismic_test_ids: {len(nonseismic_test_ids)}\n" \
                     f"source_ids: {self.source_nonseismic_len}"

        assert len(nonseismic_train_ids) + len(nonseismic_val_ids) + len(
            nonseismic_test_ids) == self.source_nonseismic_len, assert_msg

        train_ids = {"seismic": seismic_train_ids,
                     "nonseismic": nonseismic_train_ids}
        val_ids = {"seismic": seismic_val_ids,
                   "nonseismic": nonseismic_val_ids}
        test_ids = {"seismic": seismic_test_ids,
                    "nonseismic": nonseismic_test_ids}

        return train_ids, val_ids, test_ids


class H5Merger:
    """
    Unir datasets HDF5 en uno solo
    """
    def __init__(self, datasets, output):

        assert isinstance(datasets, list), "Datasets should be a list!"

        self.datasets = datasets
        self.output = output

        self.h5_out = h5py.File(self.output, 'w')

        out_seis_group = self.h5_out.create_group("earthquake/local")
        out_nonseis_group = self.h5_out.create_group("non_earthquake/noise")

        # Para cada dataset en la lista
        for dset in self.datasets:

            print(f"Merging dataset: {dset}")

            with h5py.File(dset, "r") as h5:

                grp_seis = h5["earthquake/local"]
                grp_nonseis = h5["non_earthquake/noise"]

                for arr in grp_seis:
                    data = grp_seis[arr]
                    out_seis_group.copy(data, arr)

                for arr in grp_nonseis:
                    data = grp_nonseis[arr]
                    out_nonseis_group.copy(data, arr)

        self.h5_out.close()


class H5STEAD:
    """
    Crear un dataset HDF5 más pequeño a partir del dataset STEAD
    """

    def __init__(self, stead_path, faulty_path, out_path, n_seis, n_nonseis):
        self.stead_path = stead_path
        self.faulty_path = faulty_path
        self.out_path = out_path
        self.n_seis = n_seis
        self.n_nonseis = n_nonseis

        # Leer dataset STEAD
        self.stead = h5py.File(self.stead_path, 'r')
        self.stead_seis_grp = self.stead["earthquake/local"]
        self.stead_nonseis_grp = self.stead["non_earthquake/noise"]
        self.stead_seis_len = len(self.stead_seis_grp)
        self.stead_nonseis_len = len(self.stead_nonseis_grp)

        # Lista de indices de todas las trazas disponibles en el dataset
        source_seis_ids = list(range(self.stead_seis_len))
        source_nonseis_ids = list(range(self.stead_nonseis_len))

        # Leer indices de trazas fallidas
        with open(self.faulty_path, 'r') as f:
            ln = f.readline()
            self.faulty_ids = list(map(int, ln.strip().split(',')))

        # Quitar indices de trazas fallidas del total (solo sismicas)
        source_seis_ids = list(set(source_seis_ids) - set(self.faulty_ids))

        # Seleccionar indices de trazas a copiar
        rng = default_rng()

        seis_ids_copy = rng.choice(source_seis_ids,
                                   self.n_seis, replace=False)

        nonseis_ids_copy = rng.choice(source_nonseis_ids,
                                      self.n_nonseis, replace=False)

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

        # Crear nuevo dataset hdf5
        with h5py.File(self.out_path, "w") as small_h5:

            out_seis_grp = small_h5.create_group("earthquake/local")
            out_nonseis_grp = small_h5.create_group("non_earthquake/noise")

            seismic_bar = tqdm.tqdm(total=self.n_seis, desc="Seismic traces")

            nonseismic_bar = tqdm.tqdm(total=self.n_nonseis,
                                       desc="Non Seismic traces")

            # Copiar las trazas al nuevo dataset
            for i, arr in enumerate(self.stead_seis_grp):
                if i in seis_ids_copy:
                    tr = self.stead_seis_grp[arr][:, 0]
                    tr = tr / np.amax(np.abs(tr))
                    out_seis_grp.create_dataset(arr, data=tr)
                    seismic_bar.update()

            for i, arr in enumerate(self.stead_nonseis_grp):
                if i in nonseis_ids_copy:
                    tr = self.stead_nonseis_grp[arr][:, 0]
                    tr = tr / np.amax(np.abs(tr))
                    out_nonseis_grp.create_dataset(arr, data=tr)
                    nonseismic_bar.update()

        # Cerrar los datasets
        self.stead.close()


class H5Info:
    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        self.dataset_name = self.dataset_path.split("/")[-1]

        with h5py.File(self.dataset_path, "r") as h5:
            seis_grp = h5["earthquake/local"]
            nonseis_grp = h5["non_earthquake/noise"]

            print(f'Dataset: {self.dataset_name}\n'
                  f'Seismic traces: {len(seis_grp)}\n'
                  f'Non Seismic traces: {len(nonseis_grp)}\n')


class H52Npy:
    """
    Crear dataset de entrenamiento npy desde dataset hdf5, para probar
    velocidad de entrenamiento/prueba
    """
    def __init__(self, dataset_path, dset_type):

        name = dataset_path.split("/")[-1]
        self. dataset_path = dataset_path
        self.dataset_name = name.split("hdf5")[0][:-1]
        self.dset_type = dset_type

        assert self.dset_type in ["train", "test"], "Invalid dataset type!"

        with h5py.File(self.dataset_path, "r") as h5:

            seis_grp = h5["earthquake/local"]
            nonseis_grp = h5["non_earthquake/noise"]

            seis_trs = []
            nonseis_trs = []

            # Copy seismic traces
            for dset in seis_grp:
                label = 1
                tr = seis_grp[dset][:]
                seis_trs.append(np.hstack([tr, label]))

            # Copy non seismic traces
            for dset in nonseis_grp:
                label = 0
                tr = nonseis_grp[dset][:]
                nonseis_trs.append(np.hstack([tr, label]))

        seis_trs = np.asarray(seis_trs, dtype=np.float32)
        nonseis_trs = np.asarray(nonseis_trs, dtype=np.float32)

        if seis_trs.size == 0:
            all_tr = nonseis_trs
        elif nonseis_trs.size == 0:
            all_tr = seis_trs
        else:
            all_tr = np.vstack([seis_trs, nonseis_trs])

        if dset_type == "train":
            os.makedirs("Data/TrainReady/", exist_ok=True)

            np.save(f"Data/TrainReady/{self.dataset_name}.npy",
                    all_tr.astype(np.float32))

        else:
            os.makedirs("Data/TestReady/", exist_ok=True)

            np.save(f"Data/TestReady/{self.dataset_name}.npy",
                    all_tr.astype(np.float32))


class Npy2TestReady:
    def __init__(self, dataset_path, dataset_type):

        assert dataset_type in ["seismic", "nonseismic"], "Invalid dataset type"

        self.dataset_path = dataset_path
        self.dataset_name = self.dataset_path.split("/")[-1]
        self.traces = np.load(self.dataset_path)
        self.dataset_type = dataset_type

        if self.dataset_type == "seismic":
            labels = np.ones((len(self.traces), 1))
            self.traces = np.hstack([self.traces, labels])

        else:
            labels = np.zeros((len(self.traces), 1))
            self.traces = np.hstack([self.traces, labels])

        os.makedirs("Data/TestReady/", exist_ok=True)

        np.save(f"Data/TestReady/{self.dataset_name}", self.traces)
