# from PIL import Image
import cv2
import random
import numpy as np


class Cluster(object):

    def __init__(self):
        self.pixels = []
        self.centroid = None

    def addPoint(self, pixel):
        self.pixels.append(pixel)

    def setNewCentroid(self):

        R = [colour[0] for colour in self.pixels]
        G = [colour[1] for colour in self.pixels]
        B = [colour[2] for colour in self.pixels]

        R = sum(R) / len(R)
        G = sum(G) / len(G)
        B = sum(B) / len(B)

        self.centroid = [R, G, B]
        self.pixels = []

        return self.centroid


class Kmeans(object):

    def __init__(self, k=3, max_iterations=5, min_distance=10.0, per=.25, display_per=.25):
        self.k = k
        self.max_iterations = max_iterations
        self.min_distance = min_distance
        self.per = (per, per)
        self.display_per = (display_per, display_per)

    def run(self, image):
        """
        Image: Must have channel RGBA
        """
        tmp_image = cv2.resize(image, 
                               tuple(reversed(tuple(map(int,
                                    np.array(image.shape[:2]) * self.per
                               )))), interpolation=cv2.INTER_AREA)
        tmp_image[:, :, 3][np.where(tmp_image[:, :, 3] != 0)] = 255
        self.image = image if np.where(tmp_image[:, :, 3] != 0)[0].shape[0] < self.k else tmp_image
        
#         self.image = image
        self.pixels = self.image.reshape([-1, 4])
        self.pixels = self.pixels[np.where(self.pixels[:, 3] != 0)]
        self.clusters = [None for i in range(self.k)]
#         print(self.clusters)
        self.oldClusters = None

        randomPixels = random.sample(list(self.pixels), self.k)

        for idx in range(self.k):
            self.clusters[idx] = Cluster()
            self.clusters[idx].centroid = randomPixels[idx]

        iterations = 0
#         print('Find Centroid:')

        while self.shouldExit(iterations) is False:

            self.oldClusters = [cluster.centroid for cluster in self.clusters]

#             print('Iteration: ', iterations)

            for pixel in self.pixels:
                self.assignClusters(pixel)
                
#             print([cluster.centroid for cluster in self.clusters])
#             print([len(cluster.pixels) for cluster in self.clusters])

            for cluster in self.clusters:
                if len(cluster.pixels) > 0:
#                     print('nope')
                    cluster.setNewCentroid()
#             print([cluster.centroid for cluster in self.clusters])
#             print([len(cluster.pixels) for cluster in self.clusters])

            iterations += 1
            
#         print('Cluster Pixels:')
        tmp_image = image
        
        
#         self.clusters = [cluster for cluster in self.clusters if len(cluster.pixels) > 0]
#         self.clustered = image
#         return (self.clusters)
        tmp_image = np.apply_along_axis(self.set_nearest_cluster, -1, tmp_image)
        self.clustered = cv2.resize(tmp_image, 
                               tuple(reversed(tuple(map(int,
                                        np.array(image.shape[:2]) * self.display_per
                                    )))), 
                                    interpolation = cv2.INTER_NEAREST)
        self.clustered[:, :, 3][np.where(self.clustered[:, :, 3] != 0)] = 255
        return self.clustered

    def assignClusters(self, pixel):
        shortest = float('Inf')
        for cluster in self.clusters:
            distance = self.calcDistance(cluster.centroid, pixel)
            if distance < shortest:
                shortest = distance
                nearest = cluster

        nearest.addPoint(pixel)

    def calcDistance(self, a, b):
        return np.sqrt(sum((a[:3] - b[:3]) ** 2))

    def shouldExit(self, iterations):
        flg = True

        if self.oldClusters is None:
            return False

        for idx in range(self.k):
            dist = self.calcDistance(
                np.array(self.clusters[idx].centroid),
                np.array(self.oldClusters[idx])
            )
            if dist > self.min_distance:
                flg = False

        if iterations > self.max_iterations:
            flg = True

        return flg

    def showCentroidColours(self, display_size=100):
        cent_imgs = []

        for cluster in self.clusters:
            image = np.array(list(map(int, cluster.centroid)) 
                             * (display_size ** 2)).reshape([display_size, display_size, -1])
            cent_imgs.append(image)
        return cent_imgs

    def set_nearest_cluster(self, pixel):
        if pixel[3] == 0: return pixel
        shortest = float('Inf')
        for cluster in self.clusters:
            distance = self.calcDistance(cluster.centroid, pixel[:3])
            if distance < shortest:
                shortest = distance
                nearest = np.append(cluster.centroid[:3], pixel[3])
        return nearest
