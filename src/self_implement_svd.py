import numpy as np
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import Matrix, Matrices, DenseMatrix
LINES_COUNT = 49390
WORDS_COUNT = 28990
K = 100
P = 20


raw_data = sc.textFile('tf.txt')
tf_matrix = raw_data.map(lambda i: i.split(','))

tf_matrix1 = tf_matrix.map(lambda i: i[1:])

tf_matrix2 = tf_matrix1.map(lambda i: [int(token) for token in i])

tf_matrix3 = tf_matrix2.map(lambda i: np.array(i))

tf_matrix4 = tf_matrix3.map(lambda i: (len(i), np.nonzero(i)[0], i[i!=0]))

tf_matrix5 = tf_matrix4.map(lambda i: Vectors.sparse(i[0], i[1], i[2]))

# 1min 49s , all the above process

tf_matrix5.cache()

row_tf_matrix = RowMatrix(tf_matrix5)

random_matrix = DenseMatrix(WORDS_COUNT, K+P, np.random.randn(WORDS_COUNT * (K+P))) # local matrix

Y = row_tf_matrix.multiply(random_matrix) # 38.5 s, Y is a rowmatrix(a distriuted matrix)

#  ============== QR 

Y_rdd = Y.rows  # 17.5ms
Y_rdd_with_index = Y_rdd.zipWithIndex()  # 1min 48s
Y_grouped = Y_rdd_with_index.groupBy(lambda i: i[1]/500)  # 41.1ms

def build_matris(tupl):
	iterables = tupl[1]
	result = list()
	for vec in iterables:
		ary = vec[0].toArray()
		result.append(ary)
	return np.array(result)

grouped_matrices = Y_grouped.map(build_matris)  #RDD of (500, 120) matrices, 99 matrices, 34.8us, which is 0.03ms


def cal_r(mat):
	q, r = np.linalg.qr(mat)
	return r

r_matrices = grouped_matrices.map(cal_r)  # 50.1 us, 

total_r = r_matrices.reduce(lambda i, j: np.concatenate((i, j)))  # 1 min 50 s

q, r = np.linalg.qr(total_r)  # 490 ms, r is the total result's r matrices

r_inv = np.linalg.inv(r)  # 2.16ms

R = Matrices.dense(120, 120, r_inv.flatten())  # build spark local dense matrix, 156 us

Q = Y.multiply(R)  # 25.4 ms

#  TOTAL timing : 17.5ms + 41.1ms + 0.03 + 0.05 + 1min50s + 490ms + 2.16ms + 0.156ms + 25.4ms 
#  Time: 1min50s + 576.396ms = 1min 50.576396s

#  ===============

qrdecomposition = Y.tallSkinnyQR(computeQ=True) # 1min49s

Q = qrdecomposition.Q # shape 49390 * 120
# R = qrdecomposition.R


def outer_prodcut(vec1, vec2):
	l1 = len(vec1)
	l2 = len(vec2)
	result = list()
	for element in vec1:
		result.append(vec2 * element)
	return np.array(result)

Q_rdd = Q.rows

Q_A_combine = Q_rdd.zip(tf_matrix5)

partial_matrices = Q_A_combine.map(lambda i: outer_prodcut(i[0], i[1]))

# test = partial_matrices.take(2)

B = partial_matrices.reduce(lambda i, j: i+j). # B is a 2d numpy array, shape (120, 28990). Can do normal SVD

B_svd = np.linage.svd(B)

U_b = B_svd[0]

shape = U_b.shape
U_b = U_b.flatten()
U_b = Matrices.dense(shape[0], shape[1], U_b)

U_b = Q.multiply(U_b) # update Ub using Q


# Results:
U_a = U_b.map(lambda i: i.toArray()[:K]) # take first K column
sigma_a = B_svd[1][:K]
V_a = B_svd[:, :K]


