import pydicom
import matplotlib.pyplot as plt

def view_sample_image(filename):
    dataset = pydicom.dcmread(filename)
    plt.imsave("sample.jpg", dataset.pixel_array)

#view_sample_image("sample.dcm")