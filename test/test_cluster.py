from hw2skeleton import cluster
from hw2skeleton import io
import os
import pandas as pd
import random
import itertools
import numpy as np

def test_similarity():

    def normalize_matrix(mat):
        mat = mat.subtract(min(mat.min()))
        mat = mat.divide(max(mat.max()))
        matcols = mat.divide(mat.max())
        matrows = mat.divide(mat.max(0),0)
        mat = pd.DataFrame(np.zeros((len(mat),len(mat))),
            index=mat.index,
            columns=mat.columns)
        for i in range(len(mat)):
            for j in range(len(mat)):
                mat.iloc[i,j] = np.mean([matcols.iloc[i,j],
                                        matrows.iloc[i,j]])
        return mat

    mat = pd.read_csv(os.path.join("data", "12859_2009_3124_MOESM2_ESM.mat"),sep ='\s')
    mat = normalize_matrix(mat)

    files = [f for f in os.listdir('data/') if f.endswith('.pdb')]
    simmat = pd.DataFrame(np.zeros((len(files),len(files))),columns=files,index=files)
    for a in range(len(files)):
        for b in range(a,len(files)):
            filename_a = os.path.join("data", files[a])
            filename_b = os.path.join("data", files[b])

            activesite_a = io.read_active_site(filename_a)
            activesite_b = io.read_active_site(filename_b)
            sim = cluster.compute_similarity(activesite_a, activesite_b, mat)
            simmat.iloc[a,b] = sim
            simmat.iloc[b,a] = sim
    # simmat.to_pickle('simmat10005070.pkl')

    return

def test_partition_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))
    active_sites = dict(zip(pdb_ids,active_sites))

    simmat = pd.read_pickle('simmmat_10005070.pkl')
    M, C = cluster.cluster_by_partitioning(active_sites,simmat)
    return 

def test_hierarchical_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]
    simmat = pd.read_pickle('simmmat_10005070.pkl')
    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))

    # update this assertion
    Z = cluster.cluster_hierarchically(active_sites, simmat)
    return
