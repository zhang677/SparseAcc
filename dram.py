import numpy as np
from scipy.io import mmwrite,mmread
from scipy.sparse import coo_array
from scipy import sparse

# "G": Gustavson, "O": Outerloop, "I": Innerloop
# C(i,j) = A(i,k) * B(k,j)
def linear_cache_model(dram_max,dram_min,Q,q):
  assert Q > 0 and q > 0 and dram_max > dram_min and dram_min > 0
  if q > Q:
    return dram_min
  return dram_max * (1-q/Q) + dram_min * q/Q


def get_dram_max(coo):
  '''
  [Cache A, Cache A^T, Cache C]
  '''
  A = coo.to_csr()
  A_T = A.transpose()
  (row, col) = A.shape

  if row == col:
    # B = A
    B = A
    GB = 0
    for i in range(row):
      for k in range(A.indptr[i],A.indptr[i+1]):
        A_col = A.indices[k]
        GB += B.indptr[A_col+1] - B.indptr[A_col]
    NBJ = 0
    for i in range(row):
      if A_T.indptr[i+1] == A_T.indptr[i]:
        continue
      NBJ += 1
    NAI = 0
    for i in range(row):
      if A.indptr[i+1] == A.indptr[i]:
        continue
      NAI += 1
  else:
    # B = A^T
    B = A_T
    GB = 0
    for i in range(row):
      for k in range(A.indptr[i],A.indptr[i+1]):
        A_col = A.indices[k]
        GB += B.indptr[A_col+1] - B.indptr[A_col]
    # N(B,J)
    NBJ = 0
    for i in range(row):
      if A.indptr[i+1] == A.indptr[i]:
        continue
      NBJ += 1
    # N(A,I)
    NAI = NBJ

  return {"G":[A.nnz,GB,GB+row*NAI], "O":[A.nnz,GB,GB+row*col], "I":[A.nnz*NBJ, A.nnz*NAI,0]}

def get_dram_min(coo):
  '''
  [Cache A, Cache A^T, Cache C]
  '''
  A = coo.to_csr()
  A_T = A.transpose()
  (row, col) = A.shape
  if row == col:
    # B = A
    GB = 0
    for k in range(col):
      if (A.indptr[k+1] != A.indptr[k]) and (A_T.indptr[k+1] != A_T.indptr[k]):
        GB += 1
  else:
    # B = A^T
    GB = A.nnz
    
  return {"G":[A.nnz,GB,col],"O":[A.nnz,RB,row*col],I:[A.nnz,A.nnz,0]}


def get_dram_estimate(coo, cache):
  dram_max = get_dram_max(coo)
  dram_min = get_dram_min(coo)
  G = {"max": dram_max["G"],"min": dram_min["G"],"q": cache["G"]}
  O = {"max": dram_max["O"],"min": dram_min["O"],"q": cache["O"]}
  I = {"max": dram_max["I"],"min": dram_min["I"],"q": cache["I"]}
  dram = {"G":[G["max"][0],linear_cache_model(G["max"][1],G["min"][1],G["min"][1],G["q"]),linear_cache_model(G["max"][2],G["min"][2],G["min"][2],G["q"])]}
  dram["O"] = [O["max"][0],linear_cache_model(O["max"][1],O["min"][1],O["min"][1],O["q"]),linear_cache_model(O["max"][2],O["min"][2],O["min"][2],O["q"])]
  dram["I"] = [linear_cache_model(I["max"][0],I["min"][0],I["min"][0],I["q"]),linear_cache_model(I["max"][1],I["min"][1],I["min"][1],I["q"]),I["max"][2]]
  return dram


if __name__ == "__main__":
  root = '/home/nfs_data/zhanggh/HyperGsys/HyperGsys/data/mtx_data/KDD/'
  name = 'DAWN-unique-hyperedges.mtx'
  file_path = root + name
  coo = mmread(file_path)
  dram = get_dram_estimate(coo, coo.nnz / 2)
  print(dram)