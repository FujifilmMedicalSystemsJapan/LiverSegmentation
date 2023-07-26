import argparse
import os
import tensorflow as tf
from model import LiverSegmentation
import read_write_data as read_write

import numpy as np
import pandas

from scipy.ndimage.morphology import distance_transform_edt as distance_transform

CURRENT_DIR = os.getcwd()

# Data directories
parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_dir', dest='data_dir', default= CURRENT_DIR  + '/data/', help='name of training data directory')

# Reading and preprocessing
parser.add_argument('--max_rotation', dest='max_rotation', type=float, default=22, help='rotation angle for data augmentation')
parser.add_argument('--min_ct', dest='min_ct', type=int, default=-1024, help='# min ct value')
parser.add_argument('--max_ct', dest='max_ct', type=int, default=1024, help='# max ct value')

# CNN network architecture parameters
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--image_channels', dest='image_channels', type=int, default=1, help='# channels in input image')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=7, help='# of out[ut classes')
parser.add_argument('--input_ch', dest='input_ch', type=int, default=1, help='concat vessel masks with image, portal, portal_hepatic_ivc, ')
parser.add_argument('--batch_norm', dest='batch_norm', action='store_true')
parser.add_argument('--no-batch_norm', dest='batch_norm', action='store_false')

# Training parameters
parser.add_argument('--num_epoch', dest='num_epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--base_lr', dest='base_lr', type=float, default=0.0015, help='initial learning rate for optm')
parser.add_argument('--test', dest='phase', action='store_false')
parser.add_argument('--train', dest='phase', action='store_true')
parser.add_argument('--gpu_num', dest='gpu_num', default='0', help='GPU to use for training ')

# read write model parameters
parser.add_argument('--exp_name', dest='exp_name', default='test' , help='name of experiment')
parser.add_argument('--load_model_dir', dest='load_model_dir', default=CURRENT_DIR + '/train/model_files/', help='model is loaded  from here')
parser.add_argument('--model_dir', dest='model_dir', default=CURRENT_DIR + '/train/model_files/', help='model is saved here')
parser.add_argument('--event_dir', dest='event_dir', default=CURRENT_DIR + '/train/event_files/', help='event file is saved here')
parser.add_argument('--load_from_previous', dest='load_from_previous', action='store_true')
parser.add_argument('--no-load_from_previous', dest='load_from_previous', action='store_false')

args = parser.parse_args()

def compute_metrics(gt, pred, num_classes):

    dice_list = []
    sensitivity_list = []
    specificity_list = []

    for i in range(1, num_classes + 1, 1):

        tp = np.sum(1 * np.logical_and(np.equal(pred, i), np.equal(gt, i)))
        tn = np.sum(1 * np.logical_and(np.not_equal(pred, i), np.not_equal(gt, i)))
        fp = np.sum(1 * np.logical_and(np.equal(pred, i), np.not_equal(gt, i)))
        fn = np.sum(1 * np.logical_and(np.not_equal(pred, i), np.equal(gt, i)))
        print('class', i)
        print('tp', tp)
        print('tn', tn)
        print('fp', fp)
        print('fn', fn)

        dice = (2 * tp) / (2 * tp + fp + fn + 0.01)
        sensitivity = tp / (tp + fn + 0.01)
        specificity = tn / (tn + fp + 0.01)

        dice_list.append(dice)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

    return dice_list, sensitivity_list, specificity_list

def compute_voronoi_tessellation(label_array):

    # compute voronoi tessellation
    inv = 1 * np.logical_not(np.greater(label_array, 0))
    distance_field, indices = distance_transform(inv, return_indices=True)

    size_x = np.shape(label_array)[2]
    size_y = np.shape(label_array)[1]

    flat_label_array = np.reshape(label_array, [-1])
    flat_indices = size_x * size_y * indices[0,:, :, :] + size_x * indices[1,:, :, :] + indices[2,:, :, :]
    flat_indices = np.reshape(flat_indices, [-1])

    values = np.take(flat_label_array, flat_indices)
    values = np.reshape(values, [np.shape(label_array)[0], np.shape(label_array)[1], np.shape(label_array)[2]])

    return values

def evaluate(model,sess, args, p_global_step, test_filename_list):

    test_steps = len(test_filename_list)
    for steps in range(0,test_steps,1):

        [p_filepath,
         p_total_loss,
         p_batch_images, p_liver_mask, p_portal_mask, p_hepatic_mask, p_ivc_mask,
         p_segmentation,
         p_vessel_label,
         p_dice_scores] = \
            sess.run([model.filepath,
                      model.loss,
                      model.batch_images, model.batch_liver_mask, model.batch_portal_msk, model.batch_hepatic_msk, model.batch_ivc_msk,
                      model.seg,
                      model.batch_vessel_label,
                      model.m_dice_scores],
                     feed_dict={model.is_training: False})  # [OTHERS]

        print(' total_loss:', p_total_loss)
        print('dice score', p_dice_scores)

        print('Write files')
        out_filename = p_filepath.decode("utf-8")
        basename = os.path.basename(out_filename).split('.hdr')[0] + str(p_global_step)

        # input
        p_batch_images = p_batch_images.reshape(p_batch_images.shape[1:4])
        p_liver_mask = p_liver_mask.reshape(p_batch_images.shape[0:3])
        p_portal_mask = p_portal_mask.reshape(p_batch_images.shape[0:3])
        p_hepatic_mask = p_hepatic_mask.reshape(p_batch_images.shape[0:3])
        p_ivc_mask = p_ivc_mask.reshape(p_batch_images.shape[0:3])

        # seg output
        p_segmentation = p_segmentation.reshape(np.shape(p_batch_images))

        # semantic segmentation output
        p_segmentation[np.equal(p_liver_mask, 0)] = -1
        p_segmentation[np.equal(p_ivc_mask, 1)] = -1
        out_filepath = os.getcwd() + '/prediction_image/' + args.exp_name + '/' + str(basename) + '_pred_sec.hdr'
        read_write.write_np_array_as_raw_file((p_segmentation + 1).astype(np.int16), out_filepath, x=0.7, y=0.7, z=0.7, extension='label')

        p_segmentation[np.equal(p_portal_mask,0)] = -1

        p_segmentation  = p_segmentation + 1
        p_vessel_label = p_vessel_label + 1

        out_filepath = os.getcwd() + '/prediction_image/' + args.exp_name + '/' + str(basename) + '.hdr'
        read_write.write_np_array_as_raw_file((1024*(2*p_batch_images-1)).astype(np.int16),out_filepath,x=0.7,y=0.7,z=0.7,extension='gray')

        out_filepath = os.getcwd() + '/prediction_image/' + args.exp_name + '/' + str(basename) + '_liver_msk.hdr'
        read_write.write_np_array_as_raw_file(p_liver_mask.astype(np.int16),out_filepath,x=0.7,y=0.7,z=0.7,extension='label')

        out_filepath = os.getcwd() + '/prediction_image/' + args.exp_name + '/' + str(basename) + '_portal_msk.hdr'
        read_write.write_np_array_as_raw_file(p_portal_mask.astype(np.int16),out_filepath,x=0.7,y=0.7,z=0.7,extension='label')

        out_filepath = os.getcwd() + '/prediction_image/' + args.exp_name + '/' + str(basename) + '_hepatic_ivc_msk.hdr'
        read_write.write_np_array_as_raw_file(p_hepatic_mask.astype(np.int16),out_filepath,x=0.7,y=0.7,z=0.7,extension='label')

        out_filepath = os.getcwd() + '/prediction_image/' + args.exp_name + '/' + str(basename) + '_ivc_msk.hdr'
        read_write.write_np_array_as_raw_file(p_ivc_mask.astype(np.int16),out_filepath,x=0.7,y=0.7,z=0.7,extension='label')

        out_filepath = os.getcwd() + '/prediction_image/' + args.exp_name + '/' + str(basename) + '_gt_label.hdr'
        read_write.write_np_array_as_raw_file(p_vessel_label.astype(np.int16),out_filepath,x=0.7,y=0.7,z=0.7,extension='label')

        out_filepath = os.getcwd() + '/prediction_image/' + args.exp_name + '/' + str(basename) + '_pred_label.hdr'
        read_write.write_np_array_as_raw_file(p_segmentation.astype(np.int16), out_filepath, x=0.7, y=0.7, z=0.7, extension='label')

        p_segmentation[np.less_equal(p_vessel_label,0)] = 0
        section_label = compute_voronoi_tessellation(p_segmentation)
        section_label[np.equal(p_liver_mask,0)] = 0
        section_label[np.equal(p_ivc_mask, 1)] = 0

        out_filepath = os.getcwd() + '/prediction_image/' + args.exp_name + '/' + str(basename) + '_pred_section.hdr'
        read_write.write_np_array_as_raw_file(section_label.astype(np.int16), out_filepath, x=0.7, y=0.7, z=0.7, extension='label')

        #p_vessel_label
        gt_section_label = compute_voronoi_tessellation(p_vessel_label)
        gt_section_label[np.equal(p_liver_mask, 0)] = 0
        gt_section_label[np.equal(p_ivc_mask,1)] = 0
        out_filepath = os.getcwd() + '/prediction_image/' + args.exp_name + '/' + str(basename) + '_gt_section.hdr'
        read_write.write_np_array_as_raw_file(gt_section_label.astype(np.int16), out_filepath, x=0.7, y=0.7, z=0.7, extension='label')

        # for computing the metrics
        p_segmentation[np.less_equal(p_vessel_label,0)] = 0
        dice_scores, sensitivity_scores, specificity_scores = compute_metrics(p_segmentation, p_vessel_label, args.num_classes)

        print(dice_scores)
        print(sensitivity_scores)
        print(specificity_scores)

    return

def Train(model,sess,args, train_filepaths, test_filepaths):

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver(max_to_keep=1000)

    if (args.load_from_previous):
        print('Load from previous')
        path = args.load_model_dir
        if ('Epoch' in path):
            saver.restore(sess, path)
        else:
            saver.restore(sess, tf.train.latest_checkpoint(path))
    else:
        print('Initialize randomly')
        init = tf.global_variables_initializer()
        sess.run(init)

    # Run the session multiple times to train the model_files
    p_global_step = 0

    # Read excel data
    total_steps = len(train_filepaths) * args.num_epoch

    for steps in range(0, total_steps, 1):

        [_, p_filepath, p_total_loss, p_global_step,p_dice_scores] = \
            sess.run([model.train_op, model.filepath, model.loss, model.global_step,model.m_dice_scores],
                     feed_dict={model.is_training: True})

        print('Step number', p_global_step)
        print('filepath', p_filepath)
        print('total_loss', p_total_loss)
        print('dice score', p_dice_scores)

        if (steps%len(train_filepaths) == 0):
            print('testing ', p_global_step)
            saver.save(sess, args.model_dir + args.exp_name + '/Epoch' + str(int(p_global_step)))
            evaluate(model,sess, args, p_global_step, test_filepaths)

    return

def Test(model, sess, args, test_filepaths):

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver(max_to_keep=1000)

    if('Epoch' in args.load_model_dir):
        path = args.load_model_dir
    else:
        path = tf.train.latest_checkpoint(args.load_model_dir)

    saver.restore(sess, path)

    p_global_step = int(path.rsplit('/')[-1].replace('Epoch',''))
    print('p_global_step', p_global_step)
    evaluate(model,sess, args, p_global_step, test_filepaths)

def get_filepaths(data_dir):

    filepath_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".hdr"):
                filepath = os.path.join(root, file)
                filepath_list.append(filepath)

    return filepath_list

def main(_):

    print("entered main")

    if not os.path.exists(args.data_dir):
        print('data directory does not exist')
        exit()

    if not os.path.exists(args.model_dir + args.exp_name):
        print('creating model directory to save parameters')
        os.makedirs(args.model_dir + args.exp_name)

    if not os.path.exists(args.event_dir + args.exp_name):
        print('creating event directory to view in tensorboard')
        os.makedirs(args.event_dir + args.exp_name)

    if not os.path.exists(os.getcwd() + '/prediction_image/' + args.exp_name + '/'):
        print('creating directory to save prediction image')
        os.makedirs(os.getcwd() + '/prediction_image/' + args.exp_name + '/')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        print("entered session")

        model = LiverSegmentation(args)

        train_filepaths = get_filepaths(args.data_dir + "/train/")
        test_filepaths = get_filepaths(args.data_dir + "/test/")

        print('len of test files', len(test_filepaths))
        print(test_filepaths)
        print('len of train files', len(train_filepaths))
        print(train_filepaths)

        print(args.phase)
        if args.phase:
            print('Train Phase')
            model.build(train_filepaths, test_filepaths)
            Train(model,sess,args, train_filepaths, test_filepaths)
        else:
            print('test phase')
            model.build(train_filepaths, test_filepaths)
            Test(model, sess, args, test_filepaths)

if __name__ == '__main__':
    tf.app.run()
