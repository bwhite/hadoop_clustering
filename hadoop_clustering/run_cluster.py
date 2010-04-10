import hadoopy
import random
import cPickle as pickle
import glob
import sys
VA = '/home/bwhite/projects/image_retrieval/vitrieve_algorithms/vitrieve_algorithms/'
SHARED_LIBS = glob.glob(VA + 'lib/*')

def consolidate_clusters(output, cluster_pkl):
    clusters = list(hadoopy.hdfs_cat_tb(output + '/p*'))
    clusters.sort(lambda x, y: cmp(x[0], y[0]))
    clusters = np.array([np.fromstring(x[1], dtype=np.float32) for x in clusters])
    with open(cluster_pkl, 'w') as fp:
        pickle.dump(clusters, fp, 2)

iter_cnt = 0
def main(input_path, output_path, num_clusters=1000):
    def inc_path():
        global iter_cnt
        iter_cnt +=1
        return '%s/%d' % (output_path, iter_cnt)
    def prev_path():
        return '%s/%d' % (output_path, iter_cnt)
    hadoopy.freeze(script='random_cluster.py',
                   shared_libs=SHARED_LIBS,
                   modules=['vitrieve_algorithms'],
                   remove_dir=True)
    hadoopy.run_hadoop(in_name=input_path,
                       out_name=inc_path(),
                       cmdenvs=['NUM_CLUSTERS=%d' % (num_clusters)],
                       script_path='random_cluster',
                       combiner=True,
                       frozen_path='frozen')
    consolidate_clusters(prev_path(), 'clusters.pkl')
    hadoopy.freeze(script='kmeans_cluster.py',
                   shared_libs=SHARED_LIBS,
                   modules=['vitrieve_algorithms', 'nn_c_l1',],
                   remove_dir=True)
    hadoopy.run_hadoop(in_name=prev_path(),
                       out_name=inc_path(),
                       script_path='kmeans_cluster',
                       cmdenvs=['CLUSTERS_PKL=%s' % ('clusters.pkl'),
                                 'NN_MODULE=nn_c_l1'],
                       files=['clusters.pkl'],
                       frozen_path='frozen')
    consolidate_clusters(prev_path(), 'clusters.pkl')

if __name__ == '__main__':
    prefix = str(random.random())
    print('Prefix: ' + prefix)
    main('/tmp/bwhite/output/models/0.56095992388/features', '/tmp/bwhite/output/clusters/' + prefix)
