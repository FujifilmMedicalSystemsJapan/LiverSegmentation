import numpy as np
from scipy.ndimage.interpolation import zoom
import scipy.ndimage.interpolation
import read_write_data as read_write

def get_five_segment_labels(label):

    org_label = [1,2,3,4,5,6,7,8]
    target_label = [1,2,2,3,4,]

    return

def make_pooling_compatible(image_, mask_, label_, num_pool=4):

    subtract_amount = np.remainder(image_.shape, pow(2,  num_pool))

    if (subtract_amount[0] == pow(2, num_pool)):
        subtract_amount[0] = 0
    if (subtract_amount[1] == pow(2, num_pool)):
        subtract_amount[1] = 0
    if (subtract_amount[2] == pow(2, num_pool)):
        subtract_amount[2] = 0

    z_start = int(np.floor(np.divide(subtract_amount[0], 2)))
    z_end = np.shape(image_)[0] - int(np.ceil(np.divide(subtract_amount[0], 2)))
    y_start = int(np.floor(np.divide(subtract_amount[1], 2)))
    y_end = np.shape(image_)[1] - int(np.ceil(np.divide(subtract_amount[1], 2)))
    x_start = int(np.floor(np.divide(subtract_amount[2], 2)))
    x_end = np.shape(image_)[2] - int(np.ceil(np.divide(subtract_amount[2], 2)))

    image = image_[z_start:z_end,y_start:y_end,x_start:x_end]
    mask = mask_[z_start:z_end,y_start:y_end,x_start:x_end]
    label = label_[z_start:z_end,y_start:y_end,x_start:x_end]

    return image, mask, label

def rot_around_center(image_, mask_, label_, center, max_rotation, crop_size):

    image = np.empty(np.shape(image_))
    label = np.empty(np.shape(label_))
    mask = np.empty(np.shape(mask_))

    image[:] = image_[:]
    label = label_[:]
    mask[:] = label_[:]

    # shift the reference to given center
    trans = np.identity(4)
    trans[3, 0] -= center[0]
    trans[3, 1] -= center[1]
    trans[3, 2] -= center[2]
    shift = np.transpose(trans)

    # Rotate around z axis
    theta_z_deg = max_rotation * (np.random.rand() - 0.5)
    theta_z = np.pi * (theta_z_deg / 180.0)
    rotz = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta_z), -np.sin(theta_z), 0],
        [0, np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 0, 1]
    ])

    # Rotate around y axis
    theta_y_deg = max_rotation * (np.random.rand() - 0.5)
    theta_y = np.pi * (theta_y_deg / 180.0)
    roty = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1]
    ])

    # Rotate around x axis
    theta_x_deg = max_rotation * (np.random.rand() - 0.5)
    theta_x = np.pi * (theta_x_deg / 180.0)
    rotx = np.array([
        [np.cos(theta_x), -np.sin(theta_x), 0, 0],
        [np.sin(theta_x), np.cos(theta_x), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    randint = np.random.randint(0,3)
    print('axis rand int', randint)
    rot = rotz # Default
    if randint == 0:
        rot = rotx
        print('theta_x',theta_x_deg)
    elif randint == 1:
        rot = roty
        print('theta_y',theta_y_deg)
    elif randint ==2:
        rot = rotz
        print('theta_z',theta_z_deg)
    else:
        print('something wrong with rotation')

     # Shift back to original reference
    reverse_shift = np.identity(4)
    reverse_shift[3, 0] += center[0]
    reverse_shift[3, 1] += center[1]
    reverse_shift[3, 2] += center[2]

    reverse_shift = np.transpose(reverse_shift)

    # Shift based on crop_size
    crop_shift = np.identity(4)
    crop_shift[3, 0] += np.asarray(center[0]) - np.asarray([crop_size[0]/2])
    crop_shift[3, 1] += np.asarray(center[1]) - np.asarray([crop_size[1]/2])
    crop_shift[3, 2] += np.asarray(center[2]) - np.asarray([crop_size[2]/2])

    crop_shift = np.transpose(crop_shift)

    affine = np.dot(np.dot(reverse_shift, np.dot(rot, shift)), crop_shift)
    rot_image = scipy.ndimage.interpolation.affine_transform(image, affine, output=None, order=1, mode='constant', cval=-1024,prefilter=True, output_shape=crop_size)
    rot_label = scipy.ndimage.interpolation.affine_transform(label, affine, output=None, order=0, mode='constant', cval=0, prefilter=True, output_shape=crop_size)
    rot_mask = scipy.ndimage.interpolation.affine_transform(mask, affine, output=None, order=0, mode='constant', cval=0, prefilter=True, output_shape=crop_size)

    return rot_image, rot_mask, rot_label

def assign_label(label_image, z, y, x, label, dilate_val):

    for i in range(z-dilate_val, z+dilate_val + 1, 1):
        for j in range(y-dilate_val, y+dilate_val + 1, 1):
            for k in range(x-dilate_val, x+dilate_val + 1, 1):

                label_image[i,j,k] = label

    return label_image

def calculate_bbox(msk, bit=0, z_margin = [0,0], y_margin = [0,0], x_margin = [0,0]):

    where = np.where(np.equal(np.right_shift(np.bitwise_and(msk, 1 << bit), bit), 1))
    min_z = max(min(where[0]) - z_margin[0], 0)
    max_z = min(max(where[0]) + z_margin[1], np.shape(msk)[0])
    min_y = max(min(where[1]) - y_margin[0], 0)
    max_y = min(max(where[1]) + y_margin[1], np.shape(msk)[1])
    min_x = max(min(where[2]) - x_margin[0], 0)
    max_x = min(max(where[2]) + x_margin[1], np.shape(msk)[2])

    crop_size = [max_z - min_z, max_y-min_y, max_x - min_x]
    center = np.asarray([(max_z + min_z)/2, (max_y + min_y)/2, (max_x + min_x)/2])
    bbox = [min_z, max_z, min_y, max_y, min_x, max_x]
    return center, crop_size, bbox

def get_train_images_and_label(filename, phase, num_classes):

    # read image, vessel mask and segment labels
    image, mask, label, header_info, file_corrupt = \
        read_write.read_image_mask_label(filename)

    # spacing normalization
    voxel_size = header_info[3:6]
    spacing = [0.7, 0.7, 0.7]
    image = zoom(image, ((voxel_size[0] / spacing[0]), (voxel_size[1] / spacing[1]), (voxel_size[2] / spacing[2])), order=1)
    mask = zoom(mask, ((voxel_size[0] / spacing[0]), (voxel_size[1] / spacing[1]), (voxel_size[2] / spacing[2])), order=0)
    label = zoom(label, ((voxel_size[0] / spacing[0]), (voxel_size[1] / spacing[1]), (voxel_size[2] / spacing[2])), order=0)

    # crop using liver mask
    print('image size before liver crop', np.shape(image))

    # calculate liver bounding box
    center, crop_size, bbox = calculate_bbox(mask,bit=0,z_margin=[8,8], y_margin=[8,8], x_margin=[8,8])

    # randomly rotate around liver center and crop
    if (phase == True): # rotate the image only when training
        print('rotating the image around its center')
        image, mask, label = \
            rot_around_center(image, mask, label, center, 22 , crop_size)
    else:
        image = image[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
        mask = mask[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
        label = label[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]

    print('image size after rot and crop', np.shape(image))

    # make pooling compatible
    image, mask, label, = make_pooling_compatible(image, mask, label, num_pool=4)

    print('image size after pooling compatibility', np.shape(image))

    # get different masks
    liver_msk = np.equal(np.right_shift(np.bitwise_and(mask,1<<0),0),1)
    ivc_msk = np.equal(np.right_shift(np.bitwise_and(mask,1<<1),1),1)
    portal_msk = np.equal(np.right_shift(np.bitwise_and(mask,1<<2),2),1)
    hepatic_msk = np.equal(np.right_shift(np.bitwise_and(mask, 1 << 3),3),1)

    # make value outisde liver to be zero
    image[np.equal(liver_msk,0)] = -1024

    # only consider portal mask
    label[np.equal(portal_msk, 0)] = 0

    return image.astype(np.int16), \
           liver_msk.astype(np.int16), \
           portal_msk.astype(np.int16), \
           hepatic_msk.astype(np.int16), \
           ivc_msk.astype(np.int16), \
           label.astype(np.int16)


