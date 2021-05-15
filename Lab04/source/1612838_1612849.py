# Import necessaries libraries
import csv
import numpy as np
import random
import copy
import argparse
import os
import time
import itertools
import sys

# Set random seed
# random.seed(838849)
# np.random.seed(838849)
seed = random.randrange(2**32)
random.seed(seed)
np.random.seed(seed)
print("Generated random seed. Seed is:", seed)
print()

# Define constant
EPS = 1e-8

'''
Parsing command line to get arguments
return list of argument values
'''
def parse_args():
    parser = argparse.ArgumentParser(description='K-means Clustering')
    parser.add_argument('input', type=str,
                        help='path to input file (*.csv)')
    parser.add_argument('output_model', type=str, default='model.txt',
                        help='path to output model')
    parser.add_argument('output_asgn', type=str, default='assignments.csv',
                        help='path to output assignments')
    parser.add_argument('k', type=int, default=2,
                        help='number of clusters')
    args = parser.parse_args()

    return args.input, args.output_model, args.output_asgn, args.k

'''
Get data points and headers from csv file
@param csv_dir: directory of the .csv file
return data points as numpy array and list of headers
'''
def get_data_from_csv(csv_dir):
    X = []
    with open(csv_dir, 'r') as f:
        lines = csv.reader(f)
        headers = next(lines, None)
        for line in lines:
            X.append([float(x) for x in line])
    return np.array(X), headers

'''
Generate file model.txt
@param out_dir: directory to output file
@param headers: list of headers
@param centroids: centroids generated from k-means algorithm
@param clusters: list of corresponding cluster
@param sse: sum squared errors
'''
def generate_model_file(out_dir, headers, centroids, clusters, sse):
    print('Generating model.txt...', end='')
    gen_start = time.time()

    ROW_LEN = 12
    k = len(centroids)

    sys.stdout = open(out_dir, 'w')
    print('Within cluster sum of squared errors: {}'.format(sse))
    print('Cluster centroids:')
    print(' '*(ROW_LEN + (ROW_LEN + 1)*k // 2 - len('Cluster #') // 2) + 'Cluster #')
    
    print('{1:{0}s}'.format(ROW_LEN, 'Attribute'), end=' ')
    print(' '.join(['{1:{0}d}'.format(ROW_LEN, i) for i in range(k)]))
    print('{1:{0}s}'.format(ROW_LEN, ''), end=' ')
    print(' '.join(['{1:>{0}s}'.format(ROW_LEN, '({})'.format(np.sum([clusters==i]))) for i in range(k)]))

    print('='*((ROW_LEN + 1)*(k+1)))
    for idx, header in enumerate(headers):
        print('{1:{0}s}'.format(ROW_LEN, header), end=' ')
        print(' '.join(['{1:{0}.4f}'.format(ROW_LEN, centroid[idx]) for centroid in centroids]))
    
    sys.stdout = sys.__stdout__
    print('Done, took {:6f} (s).'.format(time.time() - gen_start))

'''
Generate file assignments.csv
@param out_dir: directory to output file
@param X: data points
@param headers: list of headers
@param clusters: assignment from data point to cluster
@param sse: sum squared errors
'''
def generate_assignments_file(out_dir, X, headers, clusters):
    print('Generating assignments.csv...', end='')
    gen_start = time.time()

    sys.stdout = open(out_dir, 'w')
    print('{},Cluster'.format(','.join(headers)))
    for idx, x in enumerate(X):
        print('{},{}'.format(','.join([str(i) for i in x]), clusters[idx]))

    sys.stdout = sys.__stdout__
    print('Done, took {:6f} (s).'.format(time.time() - gen_start))

''' 
Calculate distance between points
@param d1, d2: points as k-D vectors
return distance as float
'''
def distance(d1, d2):
    return np.linalg.norm(d1 - d2)

'''
Initialize the centroids
@param X: lists of points
@param k: number of clusters
'''
def initialize_centroids(X, k, init_centroid):
    if init_centroid == 'random':
        return np.array(random.sample(list(X), k))
    elif init_centroid == 'kmeans++':
        print('-- kmeans++ --')
        c = []
        c0 = random.choice(X)
        print('Centroids #0: {}'.format(c0))
        c.append(c0)
        for i in range(1, k):
            p = np.array([np.min([distance(x, c[j]) for j in range(i)])**2 for x in X])
            p /= np.sum(p)
            ci = X[np.random.choice([i for i in range(len(X))],size=1,p=p)[0]]
            c.append(ci)

            print('Centroids #{}: {}'.format(i, ci))
        print('-- End kmeans++ --')
        return np.array(c)

'''
Calculate from current centroids the sum squared errors
@param X: data points
@param centroids: centroids
@param clusters: assignment of data point to centroid
'''
def calculate_SSE(X, centroids, clusters):
    k = len(clusters)
    sse = 0
    for idx, x in enumerate(X):
        sse += distance(x, centroids[clusters[idx]]) ** 2
    return sse

'''
Perform k-means clustering
@param X: lists of points
@param k: number of clusters
@param init_centroid: algorithm to initialize centroid ('random' or 'kmeans++')
@param max_iter: max number of iterationss
return clusters represented by centroids
'''
def kmeans(X, k, init_centroid='kmeans++', max_iter=100):
    # Assert that the number of clusters is eligible
    assert 0 < k <= len(X)
    print('-- Summary --')
    print('Size of data: {}'.format(X.shape[0]))
    print('Dimension: {}'.format(X.shape[1]))
    print('Number of clusters: {}'.format(k))
    print('-- End Summary --')
    print()

    # Initialize the points
    print('Finding initial centroid...')
    start = time.time()
    centroids_init = initialize_centroids(X, k, init_centroid)
    print('Done, took {:.6f} (s).'.format(time.time() - start))
    print('Initial centroids:\n {}'.format(centroids_init))
    print()

    # Assign points by default to the first centroids
    cluster = np.zeros(len(X)).astype('int')

    # Re-assign points until convergence
    centroids = copy.deepcopy(centroids_init)
    _iter = 0
    print('-- Begin Loop --')
    loop_start = time.time()
    diff = 1 # difference between two matrices (track convergence)
    while diff > EPS and _iter < max_iter:
        # Logging
        one_iter_start = time.time()
        print('Iteration #{}... '.format(_iter), end='')

        for idx, x in enumerate(X):
            # Calculate the distance
            distances = np.array(list(map(lambda d: distance(x, d), centroids)))
            # Assign point to the closest centroid
            cluster[idx] = np.argmin(distances)

        # Update centroids
        centroids_new = np.array([np.mean(X[cluster == i], axis=0) for i in range(k)])

        # Calculate difference from old centroids
        diff = np.linalg.norm(centroids_new - centroids)

        # Update centroids
        centroids = centroids_new

        # Logging
        print('Done, took {:.6f} (s). SSE = {:.6f}.'.format(time.time() - one_iter_start, calculate_SSE(X, centroids, cluster)))
        now = time.time()
        _iter += 1
    print('-- End Loop --')
    print('Loop took {:.6f} (s).'.format(time.time() - loop_start))
    print()

    sse_start = time.time()
    print('Calculating SSE...', end='')
    sse = calculate_SSE(X, centroids, cluster)
    print('Done, took {:.6f} (s).'.format(time.time() - sse_start))
    
    return centroids, cluster, sse

if __name__ == "__main__":
    inp, out_model, out_asgn, k = parse_args()

    assert os.path.isfile(inp)
    X, headers = get_data_from_csv(inp)

    centroids, clusters, sse = kmeans(X, k, init_centroid='kmeans++')
    generate_model_file(out_model, headers, centroids, clusters, sse)
    generate_assignments_file(out_asgn, X, headers, clusters)
