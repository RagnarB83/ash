#Standalone reorder fragments script

from functions_coords import *

coordsA=np.array([[-0.35484313,  0.0, -1.46189138],
                  [-0.35484313,  0.759337,   -0.86584838],
                  [-0.35484313, -0.759337,   -0.86584838]])

coordsB=np.array([[-2.650886895,  0.759337000, 4.853387909],
                  [-2.650886895,  -0.759337000,   4.853387909],
                  [-2.650886895, 0.000000000, 4.257344909]])

elemsA=np.array(['O','H','H'])
elemsB=np.array(['H','H','O'])

order=reorder(reorder_hungarian,coordsA,coordsB, elemsA, elemsB)
print("list:", order)

#With reflections
result_rmsd, q_swap, q_reflection, q_review = check_reflections(elemsA, elemsB, coordsA, coordsB,
                                                                reorder_method=reorder_hungarian,
                                                                rotation_method=kabsch_rmsd)

print("result_rmsd:", result_rmsd)
print("q_swap:", q_swap)
print("q_reflection:", q_reflection)
print("q_review:", q_review)

