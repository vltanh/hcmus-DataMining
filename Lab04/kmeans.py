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
# random.seed(3698)
seed = random.randrange(sys.maxsize)
rng = random.Random(seed)
print("Seed was:", seed)

# Define constant
EPS = 1e-6

'''
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
'''
def get_data_from_csv(csv_dir):
    X = []
    with open(csv_dir, 'r') as f:
        lines = csv.reader(f)
        headers = next(lines, None)
        for line in lines:
            X.append([int(x) for x in line])
    return np.array(X), headers

'''
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
'''
def generate_assignments_file(out_dir, X, headers, clusters, sse):
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
def initialize_centroids(X, k):
    return np.array(random.choices(X, k=k))

'''
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
return clusters represented by centroids
'''
def kmeans(X, k):
    # Assert that the number of clusters is eligible
    assert 0 < k <= len(X)
    print('-- Summary --')
    print('Size of data: {}'.format(X.shape[0]))
    print('Dimension: {}'.format(X.shape[1]))
    print('Number of clusters: {}'.format(k))
    print()

    # Initialize the points
    centroids_init = initialize_centroids(X, k)
    # print('+ Initial centroids:\n {}'.format(centroids_init))

    # Assign points by default to the first centroids
    cluster = np.zeros(len(X)).astype('int')

    # Re-assign points until convergence
    centroids = copy.deepcopy(centroids_init)
    _iter = 0
    print('-- Begin Loop --')
    loop_start = time.time()
    diff = 1
    while diff > EPS:
        one_iter_start = time.time()
        print('Iteration #{}... '.format(_iter), end='')
        # print('+ Current centroid:\n {}'.format(centroids))

        for idx, x in enumerate(X):
            # Calculate the distance
            distances = np.array(list(map(lambda d: distance(x, d), centroids)))
            # Assign point to the closest centroid
            cluster[idx] = np.argmin(distances)

        # Update centroids
        centroids_new = np.array([np.mean(X[cluster == i], axis=0) for i in range(k)])

        # Calculate difference from old centroids
        diff = np.linalg.norm(centroids_new - centroids)
        centroids = centroids_new

        # Logging
        print('Done, took {:.6f} (s).'.format(time.time() - one_iter_start))
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

    centroids, clusters, sse = kmeans(X, k)
    generate_model_file(out_model, headers, centroids, clusters, sse)
    generate_assignments_file(out_asgn, X, headers, clusters, sse)