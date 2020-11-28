import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def read_grayscale_image(path_to_image: str, new_size=300) -> tuple:
    image_file = Image.open(path_to_image)
    image_file = image_file.resize((new_size, new_size))
    image_gray = ImageOps.grayscale(image_file)
    
    return np.array(image_gray)

def otsu(image_path: str, n_bins=256, epsilon=0.001, intensity_range=(0, 256), white_intensity=255):
    gray_image = read_grayscale_image(image_path)
    hist = plt.hist(gray_image.ravel(), bins=n_bins, range=intensity_range, fc='k', ec='k')[0].astype(np.uint16)
    pixel_number = gray_image.shape[0] * gray_image.shape[1]

    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(n_bins)

    for threshold in range(0, n_bins):
        pcb = np.sum(hist[:threshold]) # background
        pcf = np.sum(hist[threshold:]) # foreground
        Wb = pcb / pixel_number
        Wf = pcf / pixel_number

        mu_b = np.sum(intensity_arr[:threshold] * hist[:threshold]) / (float(pcb) + epsilon)
        mu_f = np.sum(intensity_arr[threshold:] * hist[threshold:]) / (float(pcf) + epsilon)
        value = Wb * Wf * (mu_b - mu_f) ** 2

        if value > final_value:
            final_thresh = threshold
            final_value = value

    final_img = gray_image.copy()
    final_img[gray_image > final_thresh] = white_intensity
    return final_img

def segment_image(path_to_image: str):
    org  = read_grayscale_image(path_to_image)
    seg  = otsu(path_to_image)

    fig = plt.figure(figsize=(15, 15))
    fig.add_subplot(1, 3, 1)
    plt.imshow(org, cmap='gray')
    plt.title("Actual")

    fig.add_subplot(1, 3, 2)
    plt.imshow(seg, cmap='gray')
    plt.title("Otsu")
    plt.show()
    
    
segment_image("/content/drive/MyDrive/University/ComputerVision/HW8/horse.jpg")
