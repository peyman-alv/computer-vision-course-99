import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from queue import Queue

import warnings
warnings.filterwarnings("ignore")

class Pixel:
    def __init__(self, row: np.int16, column: np.int16, intensity: np.uint8):
        self.row = row
        self.column = column
        self.intensity = intensity
    
    def get_neighbours(self, im_height: np.int16, im_width: np.int16, image: np.array) -> list:
        neighbours = list()
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if (i, j) == (0, 0):
                    continue
                curr_row = self.row + i
                curr_col = self.column + j
                if 0 <= curr_row < im_height and 0 <= curr_col < im_width:
                    neighbours.append(Pixel(curr_row, curr_col, image[curr_row, curr_col]))
        return neighbours

class RegionGrowing:
    def __init__(self, image_path: str, threshold: np.uint8):
        self.image = np.array(ImageOps.grayscale(Image.open(image_path).resize((300, 300))))
        self.threshold = threshold
        
        self.height = self.width = self.image.shape[0]
        self.pixel_classes = np.zeros(shape=(self.image.shape[0], self.image.shape[1]), dtype=np.int16)
        self.pixel_classes.fill(-1)

        self.queue = Queue(maxsize=self.height * self.width)

    def BFS(self, curr_row: np.int16, curr_col: np.int16):
        if self.pixel_classes[curr_row, curr_col] == -1:
            pixel_class = np.max(self.pixel_classes) + 1
            self.pixel_classes[curr_row, curr_col] = pixel_class
            
            curr_pixel = Pixel(curr_row, curr_col, self.image[curr_row, curr_col])
            self.queue.put(curr_pixel)
            
            while not self.queue.empty():
                pixel = self.queue.get()
                neighbours = pixel.get_neighbours(self.height, self.width, self.image)
                for neighbour in neighbours:
                    if self.pixel_classes[neighbour.row, neighbour.column] == -1 and np.abs(neighbour.intensity - curr_pixel.intensity) <= self.threshold:
                     self.pixel_classes[neighbour.row, neighbour.column] = pixel_class
                     self.queue.put(neighbour)

    def determine_classes(self):
        for i in range(0, self.height):
            for j in range(0, self.width):
                self.BFS(i, j)

    def apply_segmentation(self):
        self.determine_classes()
        
        segments = np.repeat(self.pixel_classes[:, :, np.newaxis], 3, axis=2)
        for i in range(0, self.height):
            for j in range(0, self.width):
                value = segments[i, j, 0]
                if value == 1: # background
                    segments[i, j] = 255, 255, 255 # black
                else:
                    segments[i, j] = (value*42)%256 , (value*73)%256, (value*26)%256

        fig = plt.figure(figsize=(15, 15))
        fig.add_subplot(1, 3, 1)
        plt.imshow(self.image, cmap='gray')
        plt.title("Actual")

        fig.add_subplot(1, 3, 2)
        plt.imshow(segments)
        plt.title("Region Growing")
        plt.show()


a = RegionGrowing("/content/drive/MyDrive/University/ComputerVision/HW8/mri.jpg", threshold=50)
a.apply_segmentation()
