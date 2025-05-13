import numpy as np
import networkx as nx
from typing import List
from itertools import combinations
from pyseizure.data_classes.feature import Feature
from pyseizure.preprocessing.feature_engineering.correlation_feature import \
    CorrelationFeature


class GraphFeature:
    """ Works only for single channel. """

    def __init__(self,
                 raw_signal: np.array,
                 features: List[Feature] = [Feature.ECCENTRICITY],
                 frequency: int = 256,
                 single_channel: bool = True):
        self.raw_signal = raw_signal
        self.features = features
        self.frequency = frequency
        self.single_channel = single_channel
        self.binomial_coefficient = list(
            combinations(range(len(raw_signal)), 2))

        self.graph = self._create_graph()

    def calculate_features(self):
        if self.single_channel:
            result = np.array([])
            for feature in self.features:
                result = np.concatenate((result, getattr(self, feature.value)))
        else:
            result = np.array([list() for _ in range(len(self.raw_signal))])
            for feature in self.features:
                result = np.concatenate((result, getattr(self, feature.value)),
                                        axis=1)

        return result

    @property
    def eccentricity(self):
        result = dict(nx.eccentricity(self.graph, weight='weight'))
        result = np.array(list(result.values()))

        if not self.single_channel:
            return result.reshape(-1, 1)
        return result.flatten()

    @property
    def clustering_coefficient(self):
        result = dict(nx.clustering(self.graph, weight='weight'))
        result = np.array(list(result.values()))

        if not self.single_channel:
            return result.reshape(-1, 1)
        return result.flatten()

    @property
    def betweenness_centrality(self):
        result = dict(nx.betweenness_centrality(self.graph, weight='weight'))
        result = np.array(list(result.values()))

        if not self.single_channel:
            return result.reshape(-1, 1)
        return result.flatten()

    @property
    def local_efficiency(self):
        return np.array([nx.local_efficiency(self.graph)])

    @property
    def global_efficiency(self):
        return np.array([nx.global_efficiency(self.graph)])

    @property
    def diameter(self):
        return np.array(nx.diameter(self.graph, weight='weight')).flatten()

    @property
    def radius(self):
        return np.array(nx.radius(self.graph, weight='weight')).flatten()

    @property
    def characteristic_path(self):
        return np.array(
            nx.average_shortest_path_length(self.graph, weight='weight')
        ).flatten()

    def _create_graph(self):
        """
        Create graph based on imaginary part of coherence.

        Returns
        -------
        networkx.Graph
            fully connected graph with weighted edges
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(self.raw_signal)))
        for x1, x2 in self.binomial_coefficient:
            _, _, icxy = CorrelationFeature.scipy_coherence(
                self.raw_signal[x1], self.raw_signal[x2], fs=self.frequency,
                nperseg=10)
            graph.add_weighted_edges_from([(x1, x2, np.abs(np.average(icxy)))])

        return graph
