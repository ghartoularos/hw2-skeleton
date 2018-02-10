from .utils import Atom, Residue, ActiveSite
import Bio.PDB
import Bio.SeqUtils
convert = Bio.SeqUtils.IUPACData.protein_letters_3to1
import itertools
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
import pandas as pd
import random
def _uppercase_for_dict_keys(lower_dict):
    upper_dict = {}
    for k, v in lower_dict.items():
        if isinstance(v, dict):
            v = _uppercase_for_dict_keys(v)
        upper_dict[k.upper()] = v
    return upper_dict

convert = _uppercase_for_dict_keys(convert)

def compute_similarity(site_a, site_b, mat):
    """
    Compute the similarity between two given ActiveSite instances.

    Input: two ActiveSite instances
    Output: the similarity between them (a floating point number)
    """
    '''
    Normalize electrostatic similarity matrix to be between 0 and 1
    '''

    #################### Some parameters to play with ########################

    # k = increasing this increases accuracy, more random iterations 
    #     so better chance of finding optimal order
    k = 10 # this was 1000 when i generated my simmatrix

    # normfactor = increasing this increases chances of attaining high 
    #              structural similarity
    normfactor = 0.5  

    # seqsimweight = increasing this increases contirbution of sequence 
    #                similarity to overall score
    seqsimweight = 0.7

    def seq_simlarity(seq1, seq2, dict):
        score = 0
        for i in range(length):
            score += mat.loc[seq1[i],seq2[i]]
        normscore = float(score)/float(length)

        dict.update({(seq1,seq2): normscore})
        dict.update({(seq2,seq1): normscore})

        return float(score)/float(length), dict

    def norm_rmsd(site1,site2):
        # Now we initiate the superimposer:
        super_imposer = Bio.PDB.Superimposer()
        super_imposer.set_atoms(site1, site2)
        super_imposer.apply(site2)

        # RMSD:
        rmsd = super_imposer.rms
        site1vecs = np.zeros((length,3))
        site2vecs = np.zeros((length,3))
        row = 0
        for i in site1:
            site1vecs[row] = i.coord
        for i in site2:
            site2vecs[row] = i.coord

        maxrmsd = max(np.amax(pdist(site1vecs,'euclidean')),
            np.amax(pdist(site2vecs,'euclidean')))
        normrmsd = 1 - float(rmsd)/(float(maxrmsd)*normfactor)
        if normrmsd < 0:
            normrmsd = 0
        elif normrmsd > 1:
            normrmsd = 1
        return normrmsd

    # Use the first model in the pdb-files for alignment
    site_a = site_a[0]
    site_b = site_b[0]

    # Iterate of all chains in the model in order to get residues
    a_atoms = []
    b_atoms = []

    # Pull out the CA atoms from Biopython's PDB structure object
    for site in site_a, site_b:
        for chain in site:
            for res in chain:
                if site == site_a:
                    a_atoms.append(res['CA'])
                else:
                    b_atoms.append(res['CA'])
            break

    # Get number of residues in each active site
    alen = len(a_atoms)
    blen = len(b_atoms)
    length = min(alen,blen) # Going to standaridize to site with less residues

    # Initializations
    seqdict = dict()
    sims = list()
    tupes = list()
    '''
    This is where the randomness comes in. Because the order of the residues
    listed in any given active site is irrelevant, but rmsd and sequence
    similarity both require some inherent order for scoring, I've taken random
    combinations of the the list of atoms for each given active site. Presumably
    I'll find a 
    '''
    if alen == blen:
        for _ in range(k):
            set1 = np.random.permutation(a_atoms)
            set2 = np.random.permutation(b_atoms)
            tupes.append((set1, set2))
    elif alen > blen:
        for _ in range(k):
            set1 = np.random.choice(a_atoms,len(b_atoms))
            set2 = np.random.permutation(b_atoms)
            tupes.append((set1, set2))
    else:
        for _ in range(k):
            set1 = np.random.permutation(a_atoms)
            set2 = np.random.choice(b_atoms,len(a_atoms))
            tupes.append((set1, set2))

    for i in tupes:
        seq1 = ''.join([convert[j.parent.resname] for j in i[0]])
        seq2 = ''.join([convert[j.parent.resname] for j in i[1]])
        try:
            seqsim = seqdict[(seq1, seq2)]
        except:
            seqsim, seqdict = seq_simlarity(seq1, seq2, seqdict)
        structsim = norm_rmsd(i[0],i[1])
        sims.append(np.average([seqsim, structsim],weights=(seqsimweight,(1 - seqsimweight))))
    sim = max(sims)
    return sim

def cluster_by_partitioning(active_sites, simmat):
    """
    Cluster a given set of ActiveSite instances using a partitioning method.

    Input: a list of ActiveSite instances
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)

    This code was adapted from github user letiantian:
    https://github.com/letiantian/kmedoids/blob/master/kmedoids.py#L8
    I put all comments in quotes that are from the original code, and added more
    comments to show understanding of the algorithm.
    """
    # TODO make it slice according to the passed active sites
    # Pre-define the number of clusters you would like to obtain
    k = 10

    # Enfore a stop condition for iterations
    tmax = 1000

    # From the similarity matrix, obtain a distance matrix:
    D = simmat.subtract(2).multiply(-1).subtract(1).as_matrix()
    n = len(D)
    # Ensure that you haven't asked for more clusters than there are datapoints
    if k > n:
        raise Exception('Too many medoids.')
    '''
    "Initialize a unique set of valid initial cluster medoid indices since we
    can't seed different clusters with two points at the same location." 
    If we did not do this, in our shuffling of the indices to get random 
    starting medoids, we might obtain two medoids that have a distance of zero
    between them, which would make the downstream algorithm error.
    '''
    # Make sets of indices
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])

    # Find out where D == 0, if at any points
    rs , cs = np.where(D == 0)
    
    # "The rows, cols must be shuffled because we will keep the first duplicate below"
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs): # For each entry in the distance matrix with d=0:
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c) # move one to the invalid pile, keep the
                                       # the other in the valid pile

        # update the valid medoid start sites with the difference of the two sets

    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds) 
    
    if k > len(valid_medoid_inds):
        raise Exception('Too many medoids (after removing {} duplicate points).'.format(
            len(invalid_medoid_inds)))

    # "Randomly initialize an array of k medoid indices"
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k]) # Take the first k random medoid start sites

    # "Create a copy of the array of medoid indices"
    Mnew = np.copy(M)

    # "Initialize a dictionary to represent clusters"
    C = {}
    for t in range(tmax):
        # "Determine clusters, i.e. arrays of data indices"
        # Find the corresponding active sites of the medoids where D is minimized:
        J = np.argmin(D[:,M], axis=1)
        # J = for each active site, this is your closest medoid active site
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0] # Update the clusters with this info
        # "Update cluster medoids"
        # Change the medoid if there is a member within the cluster with a
        # smaller distance 
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j] # This is the new medoid
        np.sort(Mnew)
        # "Check for convergence"
        if np.array_equal(M, Mnew): # No change, don't bother clustering more
            break
        M = np.copy(Mnew)
    else:
        # "Final update of cluster memberships"
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
    return M, C

def cluster_hierarchically(active_sites, simmat):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm.

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    I tried to implement this myself, see commented code below
    Unfortunately it didn't work, so I just called a scipy function.
    """
    D = simmat.subtract(2).multiply(-1).subtract(1).as_matrix()
    # sites = list(simmat.columns)
    # print(sites)
    # clusterings = list()
    # clusterings.append([[i] for i in sites])
    # clust = list()
    # clusted = list()
    # D2 = np.copy(D)
    # for i in range(len(D)):
    #     print(i)
    #     if i in clusted:
    #         continue
    #     minind = np.argmin(D2[i][i+1:]) + i + 1
    #     clust.append([i,minind])
    #     D2 = np.delete(D2,[minind])
    #     clusted.append(i)

    # print(clust)
    # input()




    Z = linkage(squareform(D))
    Znew = np.zeros(Z.shape)
    for i in range(len(Z)):
        for k in range(4):
            Znew[i][k] = int(Z[i][k])
    Z = [list(i) for i in sorted(Znew, key = lambda x: int(x[3]))]
    # Fill in your code here!
    # print(type(Z))
    # print(Z[:10][:])

    return Z