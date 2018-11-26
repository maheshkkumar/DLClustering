import numpy as np
from keras.callbacks import Callback
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.utils.linear_assignment_ import linear_assignment


class EvaluatePerformance(object):
    def __init__(self):
        pass

    @staticmethod
    def accuracy(true_labels, predicted_labels):
        t_labels = true_labels.astype(np.int64)
        p_labels = predicted_labels

        print("Type of p_labels: {}".format(type(p_labels)))
        print("Values in p_labels: {}".format(p_labels))
        result = max(p_labels.max(), t_labels.max()) + 1
        evaluation_grid = np.zeros((result, result), dtype=np.int64)
        for size in range(p_labels.size):
            evaluation_grid[p_labels[size], t_labels[size]] += 1
        indices = linear_assignment(evaluation_grid.max() - evaluation_grid)
        return (sum([evaluation_grid[i, j] for i, j in indices]) * 1.0 / p_labels.size)

    @staticmethod
    def soft_labels_target_dist(soft_labels):
        result = (soft_labels ** 2 / soft_labels.sum(0))
        return (result.T / result.sum(1)).T


class ComputeAccuracyCallback(Callback):
    def __init__(self, data, labels, model):
        self.data = data
        self.labels = labels
        self.ae_model = model

        super(ComputeAccuracyCallback, self).__init__()

    def on_epoch_end(self, epoch, logging=None):
        if epoch % 10 == 0:
            latent_model = Model(self.ae_model.input,
                                 self.ae_model.get_layer(
                                     "ae_encoder_{}".format((int(len(self.ae_model.layers) / 2) - 1))).output)
            latent_representation = latent_model.predict(self.data)
            kmeans = KMeans(n_clusters=len(np.unique(self.labels)), n_init=20)
            labels_predicted = kmeans.fit(latent_representation)

            print("{}> Accuracy: {:.5f}, NMI: {:.5f}".format("=" * 10, EvaluatePerformance.accuracy(self.labels,
                                                                                                    labels_predicted),
                                                             normalized_mutual_info_score(self.labels,
                                                                                          labels_predicted)))
