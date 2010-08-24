import hadoopy
import random
import cPickle as pickle
import glob
import sys
import numpy as np
VA = '/home/brandyn/projects/image_retrieval/vitrieve_algorithms/vitrieve_algorithms/'
SHARED_LIBS = glob.glob(VA + 'lib/*')

def consolidate_clusters(output, cluster_pkl):
    clusters = list(hadoopy.cat(output + '/p*'))
    clusters.sort(lambda x, y: cmp(x[0], y[0]))
    clusters = np.array([np.fromstring(x[1], dtype=np.float32) for x in clusters])
    with open(cluster_pkl, 'w') as fp:
        pickle.dump(clusters, fp, 2)

def _map_cluster_canopies(cluster_canopies):
    canopy_clusters = {}
    for clust_num, canopy_nums in cluster_canopies:
        for canopy_num in canopy_nums:
            try:
                canopy_clusters[canopy_num].add(clust_num)
            except KeyError:
                canopy_clusters[canopy_num] = set((clust_num,))
    return canopy_clusters

def consolidate_canopy_clusters(output, cluster_pkl):
    clusters = list(hadoopy.cat(output + '/p*'))
    clusters.sort(lambda x, y: cmp(x[0][0], y[0][0]))
    cluster_canopies = [(x, y[0][1]) for x, y in enumerate(clusters)]
    canopy_clusters = _map_cluster_canopies(cluster_canopies)
    clusters = np.array([np.fromstring(x[1], dtype=np.float32) for x in clusters])
    with open(cluster_pkl, 'w') as fp:
        pickle.dump((clusters, canopy_clusters), fp, 2)


def gen_data(num_clusters, num_points, num_dims):
    hadoopy.launch_frozen(in_name='/tmp/bwhite/input/synth_clusters/dummy',
                          out_name='/tmp/bwhite/input/synth_clusters/%d-%d-%d' % (num_clusters, num_points, num_dims),
                          script_path='generate_data.py',
                          remove_dir=True,
                          cmdenvs=['NUM_CLUSTERS=%d' % (num_clusters),
                                   'NUM_POINTS=%d' % (num_points),
                                   'NUM_DIMS=%d' % (num_dims)],
                          #reducer=None,
                          jobconfs='mapred.reduce.tasks=30',
                          frozen_path='frozen')


def canopy(input_path, output_path, num_clusters, cluster_path, num_reducers):
    def inc_path():
        global iter_cnt
        iter_cnt +=1
        return '%s/%d' % (output_path, iter_cnt)
    def prev_path():
        return '%s/%d' % (output_path, iter_cnt)
    soft = str(4000.)
    hard = str(250.)

    hadoopy.freeze(script_path='canopy_cluster.py',
                   shared_libs=SHARED_LIBS,
                   modules=['vitrieve_algorithms', 'nn_l2sqr_c'],
                   remove_dir=True)
    hadoopy.launch(in_name=input_path,
                       out_name=inc_path(),
                       script_path='canopy_cluster.py',
                       files='nn_l2sqr.py',
                       cmdenvs=['NN_MODULE=nn_l2sqr_c',
                                'CANOPY_SOFT_DIST=%s' % (soft),
                                'CANOPY_HARD_DIST=%s' % (hard)],
                       frozen_path='frozen')
    consolidate_clusters(prev_path(), 'canopies.pkl')

    hadoopy.freeze(script_path='canopy_cluster_assign.py',
                   remove_dir=True)
    hadoopy.launch(in_name=input_path,
                       out_name=inc_path(),
                       script_path='canopy_cluster_assign.py',
                       cmdenvs=['CANOPY_SOFT_DIST=%s' % (soft),
                                'CANOPIES_PKL=' + 'canopies.pkl'],
                       files='canopies.pkl',
                       reducer=None,
                       frozen_path='frozen')
    input_path = prev_path()

    hadoopy.launch(in_name=cluster_path,
                       out_name=inc_path(),
                       script_path='canopy_cluster_assign.py',
                       cmdenvs=['CANOPY_SOFT_DIST=%s' % (soft),
                                'CANOPIES_PKL=' + 'canopies.pkl'],
                       files='canopies.pkl',
                       reducer=None,
                       frozen_path='frozen')
    consolidate_canopy_clusters(prev_path(), 'clusters.pkl')

    hadoopy.freeze(script_path='kmeans_canopy_cluster.py',
                   shared_libs=SHARED_LIBS,
                   modules=['vitrieve_algorithms', 'nn_l2sqr_c',],
                   remove_dir=True)
    hadoopy.launch(in_name=input_path,
                       out_name=inc_path(),
                       script_path='kmeans_canopy_cluster.py',
                       cmdenvs=['CLUSTERS_PKL=%s' % ('clusters.pkl'),
                                'CANOPY_SOFT_DIST=%s' % (soft),
                                 'NN_MODULE=nn_l2sqr_c'],
                       files=['nn_l2sqr_c.py', 'clusters.pkl'],
                       frozen_path='frozen')



def random_cluster(input_path, output_path, num_clusters, cluster_path, num_reducers):
    def inc_path():
        global iter_cnt
        iter_cnt +=1
        return '%s/%d' % (output_path, iter_cnt)
    hadoopy.freeze(script_path='random_cluster.py',
                   shared_libs=SHARED_LIBS,
                   modules=['vitrieve_algorithms'],
                   remove_dir=True)
    hadoopy.launch(in_name=input_path,
                       out_name=inc_path(),
                       cmdenvs=['NUM_CLUSTERS=%d' % (num_clusters)],
                       script_path='random_cluster.py',
                       #combiner=True,
                       frozen_path='frozen')

 


iter_cnt = -1
def main(input_path, output_path, num_clusters, cluster_path, num_reducers):
    def inc_path():
        global iter_cnt
        iter_cnt +=1
        return '%s/%d' % (output_path, iter_cnt)
    def prev_path():
        return '%s/%d' % (output_path, iter_cnt)
    consolidate_clusters(cluster_path, 'clusters.pkl')
    if 1:
        hadoopy.launch_frozen(in_name=input_path,
                              out_name=inc_path(),
                              script_path='kmeans_cluster_single.py',
                              reducer=None,
                              cmdenvs=['CLUSTERS_PKL=%s' % ('clusters.pkl'),
                                       'NN_MODULE=nn_l2sqr_c'],
                              #combiner=True,
                              files=['nn_l2sqr_c.py','clusters.pkl'],
                              shared_libs=SHARED_LIBS,
                              modules=['vitrieve_algorithms', 'nn_l2sqr_c',],
                              remove_dir=True,
                              jobconfs=['mapred.min.split.size=999999999999',
                                        'mapred.reduce.tasks=%d' % (num_reducers)])
                                        #'mapred.map.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec',
                                        #'mapred.map.output.compression.type=BLOCK',
                                        #'mapred.compress.map.output=true'])
        #consolidate_clusters(prev_path(), 'clusters.pkl')
 
if __name__ == '__main__':
    for run_num in range(1):
        prefix = str(random.random())
        print('Prefix: ' + prefix)
        iter_cnt = -1
        dat = [[0, '100-20-1000', 100, 1],
               [1, '100-110-1000', 100, 1], 
               [2, '100-200-1000', 100, 1],
               [3, '100-1100-1000', 100, 1],
               [4, '100-2000-1000', 100, 1]]#, 
               #[5, '100-11000-1000', 100, 1],
               #[6, '100-20000-1000', 100, 1]]
        for x, y, z, q in dat:
            #main('/tmp/bwhite/input/synth_clusters/' + y, '/tmp/bwhite/output/clusters/' + prefix, z, '/tmp/bwhite/output/clusters/0.991472772397/' + str(x), q)
            main('/tmp/bwhite/output/clusters/0.252632615449/' + str(x), '/tmp/bwhite/output/clusters/' + prefix, z, '/tmp/bwhite/output/clusters/0.991472772397/' + str(x), q)
            

    #gen_data(100, 20000, 1000)
    #gen_data(100, 11000, 1000)
    #gen_data(100, 1100, 1000)
    #gen_data(100, 110, 1000)
    #gen_data(100, 20, 1000) # 400 Meg
    #gen_data(100, 200, 1000) # 4 Gig
    #gen_data(100, 2000, 1000) # 40 Gig
    #gen_data(1000, 2, 1000) # 400 Meg
    #gen_data(1000, 20, 1000) # 4 Gig
    #gen_data(1000, 200, 1000) # 40 Gig

