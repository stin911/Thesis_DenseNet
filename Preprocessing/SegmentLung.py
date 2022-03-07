import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \
    reconstruction, binary_closing
from skimage.measure import label, regionprops, perimeter
from skimage.filters import roberts, sobel
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import pathlib
import SimpleITK as sitk
import argparse
from pathlib import Path


def threshold_based_crop(image, image2):
    """
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box.
    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.
        :param image:
        :param image2:
    """
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    inside_value = 0
    outside_value = 255
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(sitk.OtsuThreshold(image, inside_value, outside_value))
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size

    return sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box) / 2):],
                                 bounding_box[0:int(len(bounding_box) / 2)]), sitk.RegionOfInterest(image2,
                                                                                                    bounding_box[int(
                                                                                                        len(
                                                                                                            bounding_box) / 2):],
                                                                                                    bounding_box[0:int(
                                                                                                        len(
                                                                                                            bounding_box) / 2)])


def get_segmented_lungs(im, mask, threshold):
    """

    :param threshold:
    :param im: 2D slice
    :param mask: boolean : True:return binary mask /False return lung mask
    :return:
    """
    # Convert into a binary image.
    binary = im < threshold
    plt.imshow(binary, cmap=plt.gray())
    # Remove the blobs connected to the border of the image
    cleared = clear_border(binary)
    # Label the image
    label_image = label(cleared)
    # Keep the labels with 2 largest areas
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    # Closure operation with disk of radius 12
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    selem = disk(10)
    binary = binary_closing(binary, selem)
    # Fill in the small holes inside the lungs
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    # Superimpose the mask on the input image
    if mask:
        binary = ndi.binary_fill_holes(edges)
        bin2 = (binary.astype(int))
        return bin2
    else:
        get_high_vals = binary == 0
        im[get_high_vals] = 0

    return im


def Seg(file, in_dire, out_dire, binary, threshold):
    """

    :param file:
    :param in_dire:
    :param out_dire:
    :param binary:
    :param threshold:
    :return:
    """
    ct_scan1 = sitk.ReadImage(in_dire + file)
    ct_scan = sitk.GetArrayFromImage(ct_scan1)
    segmented_ct_scan = segment_lung_from_ct_scan(ct_scan, mask=binary, threshold=threshold)
    xs, ys, zs = np.where(segmented_ct_scan != 0)
    segmented_ct_scan = segmented_ct_scan[min(xs):max(xs) + 1, min(ys):max(ys) + 1, min(zs):max(zs) + 1]
    im = sitk.GetImageFromArray(segmented_ct_scan)
    im.SetOrigin(ct_scan1.GetOrigin())
    im.SetSpacing(ct_scan1.GetSpacing())
    sitk.WriteImage(im, out_dire + file)


def SegM(file, in_dire, out_dire, binary, threshold):
    """

    :param file:image
    :param in_dire:directory of file
    :param out_dire: destination directory
    :param binary: boolean
    :param threshold: treshold
    :return:
    """
    ct_scan1 = sitk.ReadImage(in_dire + file)
    ct_scan = sitk.GetArrayFromImage(ct_scan1)
    segmented_ct_scan = segment_lung_from_ct_scan(ct_scan, mask=binary, threshold=threshold)
    im = sitk.GetImageFromArray(segmented_ct_scan)
    im.SetOrigin(ct_scan1.GetOrigin())
    im.SetSpacing(ct_scan1.GetSpacing())
    sitk.WriteImage(im, out_dire + file)

# CROP the original image using the generated mask
def WriteCropImage(dir_path, mask_path, output_dir):
    """
    Write the cropped image based on the mask
    :param output_dir: output directory
    :param dir_path: path to the images
    :param mask_path: path to the masks

    """
    for file in os.listdir(dir_path):
        print(file)
        image = sitk.ReadImage(dir_path + file)
        mask = sitk.ReadImage(mask_path + file )
        crop_mask, crop_image = threshold_based_crop(mask, image)
        sitk.Show(crop_image)
        sitk.WriteImage(crop_image, output_dir + file)


def segment_lung_from_ct_scan(ct_scan, mask, threshold):
    """

    :param ct_scan:
    :param mask:
    :param threshold:
    :return:
    """
    return np.asarray([get_segmented_lungs(slice, mask, threshold) for slice in ct_scan])


class SegmentLung:
    def __init__(self, directory, destination, threshold, binary):
        self.directory = directory
        self.destination = destination
        self.threshold = threshold
        self.binary = binary

    def segment(self):
        """

        :return:
        """
        for filename in os.listdir(self.directory):
            if self.binary:
                SegM(filename, in_dire=self.directory, out_dire=self.destination,
                     binary=self.binary, threshold=self.threshold)
            else:
                Seg(filename, in_dire=self.directory, out_dire=self.destination,
                    binary=self.binary, threshold=self.threshold)


if __name__ == '__main__':
    # If you use it from cmd
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Path to folder on local disk", type=Path)
    parser.add_argument("--mask_dest", help="Path to destination on local disk", type=Path)
    parser.add_argument("--bounding_dest", help="Bounding box destination folder", type=path)
    parser.add_argument("--threshold", default=-400, help="Set the threshold for the segmentation", type=int)
    parser.add_argument("--binary", default=False, help="return binary mask", type=bool, required=False)
    

    args = parser.parse_args()
    test = SegmentLung(args.directory, args.mask_dest, args.threshold, args.binary)
    test.segment()
    WriteCropImage(args.directory, args.mask_dest, args.bounding_dest)
    
    # If you use it in your IDE
    """test = SegmentLung("FOLDER_TO_THE_IMAGE", "WHERE_STORE_THE_MASK"", "HU_VALUE", "TRUE") # SET TRUE if you want a binary mask ,FALSe if you want the original image segmented
    test.segment()
    WriteCropImage("D:/NrrdSameSp/", "D:/MaskBMS/", "D:/BoundingBoxBMS/")"""
