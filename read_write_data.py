import numpy as np
import time
import os

def read_image_mask_label(filename):

    filename = (filename.decode("utf-8")).rsplit(".hdr", 1)[0]

    # Check whether files exist in the directed directory
    if not(os.path.isfile(filename + '.raw') and os.path.isfile(filename + '.msk') and os.path.isfile(filename + '.label')):
        print("The corresponding files does not exist")
        return [], [], [], [], True

    # Get image info from header file
    header_data = np.loadtxt(filename + '.hdr', delimiter=' ', usecols=range(7))
    header_info = [int(header_data[2]), int(header_data[1]), int(header_data[0]), header_data[6], header_data[5], header_data[4]]

    # To measure reading time
    print('reading images and mask file')
    start_time = time.time()

    # Try to read the file or else throw error and return
    try:
        image = np.fromfile(filename + '.raw', dtype=np.int16)
        mask = np.fromfile(filename + '.msk', dtype=np.uint16)
        label = np.fromfile(filename + '.label', dtype=np.uint16)
    except:
        print('Some problem while reading file')
        return [], [], [], [], True

    print('total reading time:', time.time() - start_time)

    # Reject corrupt file or else continue to pre-processing step
    if ((np.shape(mask) < header_data[2] * header_data[1] * header_data[0]) or
            (np.shape(image) < header_data[2] * header_data[1] * header_data[0])):
        print('Image and mask size do not match with dimensions in header file')

        return [], [], [], [], True

   # RESHAPE image and mask data
    image = np.reshape(image, header_info[0:3])
    mask = np.reshape(mask, header_info[0:3])
    label = np.reshape(label, header_info[0:3])

    return image, mask, label, header_info, False

def write_np_array_as_raw_file(array, filename, x = 1.0, y = 1.0, z = 1.0, extension = 'label'):

    # FileSize = np.array([array.shape[2], array.shape[1], array.shape[0]], dtype=np.int16);
    # np.savetxt(filename.rsplit(".", 1)[0] + '.hdr', FileSize.astype(int), delimiter=' , ', newline='\n',fmt='%1.1f')
    with open(filename.rsplit(".", 1)[0] + '.hdr', 'w') as f:
        f.write(str(array.shape[2]) + ' ' + str(array.shape[1]) + ' ' + str(array.shape[0]) + ' ' + str(2) + ' ' +
                str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + extension)

    array.tofile(filename.rsplit(".", 1)[0] + '.raw')

    return

