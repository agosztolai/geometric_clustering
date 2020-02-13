'''Functions for saving and loading'''
import pickle as pickle


def save_curvatures(times, kappas, filename = 'curvature'):
    '''save curvatures in a pickle'''
    pickle.dump([times, kappas], open(filename + '.pkl', 'wb'))


def load_curvature(filename = 'curvature'):
    '''load curvatures from a pickle'''
    times, kappas = pickle.load(open(filename + '.pkl', 'rb'))
    return times, kappas 




def save_clustering(self, filename = None):
    if not filename:
        filename = self.G.graph.get('name')
    pickle.dump([self.G, self.clustering_results, self.labels_gt], open(filename + '_cluster_' + self.cluster_tpe + '.pkl','wb'))


def load_clustering(self, filename = None):
    if not filename:
        filename = self.G.graph.get('name')
        
    self.G, self.clustering_results, self.labels_gt = pickle.load(open(filename + '_cluster_' + self.cluster_tpe + '.pkl','rb'))


def save_embedding(self, filename = None):
    pickle.dump([self.G, self.Y], open(filename + '_embed.pkl','wb'))


