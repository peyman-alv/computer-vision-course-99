import cv2
import numpy as np
import matplotlib.pyplot as plt

class NMS():
    def __read_image(self, img_path: str) -> np.array:
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    def __apply_gaussian_blur(self, img: np.array) -> np.array:
        return cv2.GaussianBlur(img, (3, 3), 0)

    def __apply_sobel_filter(self, img: np.array) -> tuple:
        convolved = np.zeros(img.shape)
        Gx, Gy = np.zeros(img.shape), np.zeros(img.shape)
        kernel_x = np.array(([-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]))
        
        kernel_y = np.array(([-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1])) 

        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                Gx[i, j] = np.sum(np.multiply(img[i - 1 : i + 2, j - 1 : j + 2], kernel_x))
                Gy[i, j] = np.sum(np.multiply(img[i - 1 : i + 2, j - 1 : j + 2], kernel_y))
    
        convolved = np.sqrt(np.square(Gx) + np.square(Gy))
        convolved = np.multiply(convolved, 255.0 / convolved.max())

        angles = np.rad2deg(np.arctan2(Gy, Gx))
        angles[angles < 0] += 180
        convolved = convolved.astype('uint8')
    
        return convolved, angles
    
    def __apply_NMS(self, img: np.array, angles: np.array) -> np.array:
        suppressed = np.zeros(img.shape)
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                    value_to_compare = max(img[i, j - 1], img[i, j + 1])
                elif (22.5 <= angles[i, j] < 67.5):
                    value_to_compare = max(img[i - 1, j - 1], img[i + 1, j + 1])
                elif (67.5 <= angles[i, j] < 112.5):
                    value_to_compare = max(img[i - 1, j], img[i + 1, j])
                else:
                    value_to_compare = max(img[i + 1, j - 1], img[i - 1, j + 1])
                
                if img[i, j] >= value_to_compare:
                    suppressed[i, j] = img[i, j]

        return np.multiply(suppressed, 255.0 / suppressed.max())

    def __img_show(self, img1: np.array, title1: str, img2: np.array, title2: str):
        fig = plt.figure(figsize=(15, 15))
        fig.add_subplot(1, 3, 1)
        plt.imshow(img1, cmap='gray')
        plt.title(title1)

        fig.add_subplot(1, 3, 2)
        plt.imshow(img2, cmap='gray')
        plt.title(title2)

    def __call__(self, img_path):
        img = self.__read_image(img_path)
        img = self.__apply_gaussian_blur(img)
        convolved, angles = self.__apply_sobel_filter(img)
        edges = self.__apply_NMS(convolved, angles)
        self.__img_show(img, 'Orginal', edges, 'NMS')
        
if __name__== "__main__":
  NMS()('path_to_your_image')
