from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ops import *
import input as input

class LiverSegmentation(object):

    def __init__(self, args):

        self.num_classes = args.num_classes
        self.phase = args.phase
        self.min_ct = args.min_ct
        self.max_ct = args.max_ct
        self.input_ch = args.input_ch
        self.batch_size = 1
        self.base_lr = args.base_lr
        self.is_batch_stats = True

    def parser(self, filepath):

        batch_images, batch_liver_mask, batch_portal_msk, batch_hepatic_msk, batch_ivc_msk, batch_vessel_label = \
            tf.py_func(input.get_train_images_and_label,
                       [filepath, self.phase, self.num_classes],
                       [tf.int16, tf.int16, tf.int16, tf.int16, tf.int16, tf.int16])

        # normal input image grey scale values
        shift = -(self.min_ct)
        scale = (self.max_ct + shift)
        print('shift', shift)
        print('scale', scale)
        batch_images = tf.clip_by_value(tf.divide(np.add(batch_images, shift), scale), 0, 1)

        # create concatenated input data
        batch_images = tf.cast(tf.reshape(batch_images,
                                          [tf.shape(batch_images)[0], tf.shape(batch_images)[1],
                                           tf.shape(batch_images)[2], 1]), dtype=tf.float32)
        batch_liver_mask = tf.cast(tf.reshape(batch_liver_mask,
                                              [tf.shape(batch_liver_mask)[0], tf.shape(batch_liver_mask)[1],
                                               tf.shape(batch_liver_mask)[2], 1]), dtype=tf.float32)
        batch_portal_msk = tf.cast(tf.reshape(batch_portal_msk,
                                              [tf.shape(batch_portal_msk)[0], tf.shape(batch_portal_msk)[1],
                                               tf.shape(batch_portal_msk)[2], 1]), dtype=tf.float32)
        batch_hepatic_msk = tf.cast(tf.reshape(batch_hepatic_msk,
                                               [tf.shape(batch_hepatic_msk)[0], tf.shape(batch_hepatic_msk)[1],
                                                tf.shape(batch_hepatic_msk)[2], 1]), dtype=tf.float32)
        batch_ivc_msk = tf.cast(tf.reshape(batch_ivc_msk,
                                           [tf.shape(batch_ivc_msk)[0], tf.shape(batch_ivc_msk)[1],
                                            tf.shape(batch_ivc_msk)[2], 1]), dtype=tf.float32)

        # concatenate image, liver mask, portal mask, hepatic mask
        batch_input_data = tf.concat([batch_images, batch_liver_mask, batch_portal_msk, batch_hepatic_msk], axis=3)

        # reshape in tensorflow format
        batch_images =tf.cast(tf.reshape(batch_images,
                                   [1, tf.shape(batch_images)[0],tf.shape(batch_images)[1],tf.shape(batch_images)[2], 1]), dtype=tf.float32)
        batch_input_data =  tf.reshape(batch_input_data,
                                   [1, tf.shape(batch_input_data)[0],tf.shape(batch_input_data)[1],tf.shape(batch_input_data)[2], self.input_ch])

        # batch_portal_msk = tf.reshape(batch_portal_msk,
        #                            [tf.shape(batch_portal_msk)[0],tf.shape(batch_portal_msk)[1],tf.shape(batch_portal_msk)[2]])
        batch_vessel_label = tf.reshape(batch_vessel_label,
                                        [tf.shape(batch_vessel_label)[0],tf.shape(batch_vessel_label)[1],tf.shape(batch_vessel_label)[2]])


        batch_vessel_label = batch_vessel_label - 1

        return batch_images, batch_input_data, batch_vessel_label, \
               batch_liver_mask, batch_portal_msk, batch_hepatic_msk, batch_ivc_msk, \
               filepath

    def get_train_iterator(self, filepaths):

        ds = tf.data.Dataset.from_tensor_slices(filepaths)
        num_files = tf.cast(tf.shape(filepaths)[0], tf.int64)
        ds = ds.shuffle(num_files)
        ds = ds.map(self.parser, num_parallel_calls=10)
        ds = ds.repeat()
        ds = ds.prefetch(1)
        iterator = ds.make_one_shot_iterator()
        return iterator

    def get_test_iterator(self, filepaths):

        ds = tf.data.Dataset.from_tensor_slices(filepaths)
        ds = ds.map(self.parser)
        ds = ds.repeat()
        ds = ds.prefetch(1)
        iterator = ds.make_one_shot_iterator()

        return iterator

    def get_data_from_iterator(self, iterator):

        batch_images, batch_input_data, batch_vessel_label, \
        batch_liver_mask, batch_portal_msk, batch_hepatic_msk, batch_ivc_msk, \
        filepath = iterator.get_next()

        return batch_images, batch_input_data, batch_vessel_label, \
               batch_liver_mask, batch_portal_msk, batch_hepatic_msk, batch_ivc_msk, \
               filepath

    def build(self, train_filepaths, test_filepaths):

        self.is_training = tf.placeholder(dtype=tf.bool, shape=[])
        self.train_iterator = self.get_train_iterator(train_filepaths)
        self.test_iterator = self.get_test_iterator(test_filepaths)

        self.batch_images, self.batch_input_data, self.batch_vessel_label, \
        self.batch_liver_mask, self.batch_portal_msk, self.batch_hepatic_msk, self.batch_ivc_msk, \
        self.filepath \
            = tf.cond(self.is_training, lambda: self.get_data_from_iterator(self.train_iterator),
                      lambda: self.get_data_from_iterator(self.test_iterator))

        with tf.name_scope('Inference') as scope:
            print('input data shape', tf.shape(self.batch_input_data))
            self.logits = self.inference(self.batch_input_data, self.num_classes)
            self.prob_per_class = tf.nn.softmax(self.logits)
            self.seg = tf.argmax(self.logits, 4)

        with tf.name_scope('Loss') as scope:
            self.loss = self.dice_loss5(self.prob_per_class, self.batch_vessel_label, self.num_classes)

        with tf.name_scope('TrainOp') as scope:
            self.train_op = self.TrainOpAdamWithoutEstimator(self.loss)

        return

    def inference(self, batch_images, num_classes):

        with tf.variable_scope('Conv1') as scope:

            with tf.variable_scope('Conv1_1') as scope:
                bias_out = conv3d_xavier(batch_images, 8)
                bn_out = batch_norm(bias_out,self.is_batch_stats)
                self.Conv1_1_out = tf.nn.relu(bn_out, name='ReLU')

            with tf.variable_scope('Conv1_2') as scope:

                bias_out =  conv3d_xavier(self.Conv1_1_out,8)
                bn_out = batch_norm(bias_out,self.is_batch_stats)
                self.Conv1_2_out = tf.nn.relu(bn_out, name='ReLU')

        with tf.name_scope('Pool1') as scope:
            self.Pool1_out = tf.nn.max_pool3d(self.Conv1_2_out, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool')

        with tf.variable_scope('Conv2') as scope:

            with tf.variable_scope('Conv2_1') as scope:

                bias_out =  conv3d_xavier(self.Pool1_out,16)
                bn_out = batch_norm(bias_out,self.is_batch_stats)
                self.Conv2_1_out = tf.nn.relu(bn_out, name='ReLU')

            with tf.variable_scope('Conv2_2') as scope:

                bias_out =  conv3d_xavier(self.Conv2_1_out,16)
                bn_out = batch_norm(bias_out,self.is_batch_stats)
                self.Conv2_2_out = tf.nn.relu(bn_out, name='ReLU')

        with tf.name_scope('Pool2') as scope:
            Pool2_out = tf.nn.max_pool3d(self.Conv2_2_out, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pooling')

        with tf.variable_scope('Conv3') as scope:

            with tf.variable_scope('Conv3_1') as scope:

                bias_out =  conv3d_xavier(Pool2_out,32)
                bn_out = batch_norm(bias_out,self.is_batch_stats)
                self.Conv3_1_out = tf.nn.relu(bn_out, name='ReLU')

            with tf.variable_scope('Conv3_2') as scope:

                bias_out =  conv3d_xavier(self.Conv3_1_out,32)
                bn_out = batch_norm(bias_out,self.is_batch_stats)
                self.Conv3_2_out = tf.nn.relu(bn_out, name='ReLU')

        with tf.name_scope('Pool3') as scope:
            self.Pool3_out = tf.nn.max_pool3d(self.Conv3_2_out, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pooling')

        with tf.variable_scope('Conv4') as scope:

            with tf.variable_scope('Conv4_1') as scope:

                bias_out =  conv3d_xavier(self.Pool3_out,64)
                bn_out = batch_norm(bias_out,self.is_batch_stats)
                self.Conv4_1_out = tf.nn.relu(bn_out, name='ReLU')

            with tf.variable_scope('Conv4_2') as scope:

                bias_out =  conv3d_xavier(self.Conv4_1_out,64)
                bn_out = batch_norm(bias_out,self.is_batch_stats)
                self.Conv4_2_out = tf.nn.relu(bn_out, name='ReLU')

            # if (self.insert_dropout == True):
            #     print('inserting dropout')
            #     self.Conv4_2_out = tf.nn.dropout(self.Conv4_2_out, keep_prob=0.5, name='dropout1')

        with tf.name_scope('Pool4') as scope:
            self.Pool4_out = tf.nn.max_pool3d(self.Conv4_2_out, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pooling')

        with tf.variable_scope('Conv5') as scope:

            with tf.variable_scope('Conv5_1') as scope:
                bias_out = conv3d_xavier(self.Pool4_out, 128)
                bn_out = batch_norm(bias_out, self.is_batch_stats)
                self.Conv5_1_out = tf.nn.relu(bn_out, name='ReLU')

                # if (mode == tf.estimator.ModeKeys.TRAIN):
                self.Conv5_1_out = tf.compat.v2.nn.dropout(self.Conv5_1_out, rate=0.3)

            with tf.variable_scope('Conv5_2') as scope:
                bias_out = conv3d_xavier(self.Conv5_1_out, 128)
                bn_out = batch_norm(bias_out, self.is_batch_stats)

                self.Conv5_2_out = tf.nn.relu(bn_out, name='ReLU')

                # if (mode == tf.estimator.ModeKeys.TRAIN):
                self.Conv5_2_out = tf.compat.v2.nn.dropout(self.Conv5_2_out, rate=0.3)

        with tf.variable_scope('Upsample1') as scope:

            size1 = tf.to_int32(tf.divide(tf.shape(batch_images), 8))
            size1 = tf.multiply(size1, tf.constant([0, 1, 1, 1, 0]));
            size1 = tf.add(size1, tf.constant([self.batch_size, 0, 0, 0, 128]))

            self.Upsample1_out = deconv3d_trilinear(self.Conv5_2_out,128,size1)

        with tf.variable_scope('Fuse1') as scope:
            with tf.variable_scope('Concatenate') as scope:
                self.Fuse1_out = tf.concat([self.Conv4_2_out, self.Upsample1_out], 4)

        with tf.variable_scope('Uconv4') as scope:

            with tf.variable_scope('UConv4_1') as scope:

                bias_out =  conv3d_xavier(self.Fuse1_out,64)
                bn_out = batch_norm(bias_out,self.is_batch_stats)
                self.UConv4_1_out = tf.nn.relu(bn_out, name='ReLU')

            with tf.variable_scope('UConv4_2') as scope:
                bias_out = conv3d_xavier(self.UConv4_1_out, 32)
                bn_out = batch_norm(bias_out, self.is_batch_stats)
                self.UConv4_2_out = tf.nn.relu(bn_out, name='ReLU')

        with tf.variable_scope('Upsample2') as scope:

            size2 = tf.to_int32(tf.divide(tf.shape(batch_images), 4))
            size2 = tf.multiply(size2, tf.constant([0, 1, 1, 1, 0]));
            size2 = tf.add(size2, tf.constant([self.batch_size, 0, 0, 0, 32]))

            self.Upsample2_out = deconv3d_trilinear(self.UConv4_2_out,32,size2)

        with tf.variable_scope('Fuse2') as scope:
            with tf.variable_scope('Concatenate') as scope:
                self.Fuse2_out = tf.concat([self.Conv3_2_out, self.Upsample2_out], 4)

        with tf.variable_scope('Uconv3') as scope:

            with tf.variable_scope('UConv3_1') as scope:

                bias_out =  conv3d_xavier(self.Fuse2_out,32)
                bn_out = batch_norm(bias_out,self.is_batch_stats)
                self.UConv3_1_out = tf.nn.relu(bn_out, name='ReLU')

            with tf.variable_scope('UConv3_2') as scope:
                bias_out = conv3d_xavier(self.UConv3_1_out, 16)
                bn_out = batch_norm(bias_out, self.is_batch_stats)
                self.UConv3_2_out = tf.nn.relu(bn_out, name='ReLU')

        with tf.variable_scope('Upsample3') as scope:

            size3 = tf.to_int32(tf.divide(tf.shape(batch_images), 2))
            size3 = tf.multiply(size3, tf.constant([0, 1, 1, 1, 0]));
            size3 = tf.add(size3, tf.constant([self.batch_size, 0, 0, 0, 16]))

            self.Upsample3_out = deconv3d_trilinear(self.UConv3_2_out,16,size3)

        with tf.variable_scope('Fuse3') as scope:
            with tf.variable_scope('Concatenate') as scope:
                self.Fuse3_out = tf.concat([self.Conv2_2_out, self.Upsample3_out], 4)

        with tf.variable_scope('Uconv2') as scope:

            with tf.variable_scope('UConv2_1') as scope:

                bias_out =  conv3d_xavier(self.Fuse3_out,16)
                bn_out = batch_norm(bias_out,self.is_batch_stats)
                self.UConv2_1_out = tf.nn.relu(bn_out, name='ReLU')

            with tf.variable_scope('UConv2_2') as scope:
                bias_out = conv3d_xavier(self.UConv2_1_out, 8)
                bn_out = batch_norm(bias_out, self.is_batch_stats)
                self.UConv2_2_out = tf.nn.relu(bn_out, name='ReLU')

        with tf.variable_scope('Upsample4') as scope:

            size3 = tf.to_int32(tf.divide(tf.shape(batch_images), 1))
            size3 = tf.multiply(size3, tf.constant([0, 1, 1, 1, 0]));
            size3 = tf.add(size3, tf.constant([self.batch_size, 0, 0, 0, 8]))

            self.Upsample4_out = deconv3d_trilinear(self.UConv2_2_out,8,size3)

        with tf.variable_scope('Fuse4') as scope:
            with tf.variable_scope('Concatenate') as scope:
                self.Fuse4_out = tf.concat([self.Conv1_2_out, self.Upsample4_out], 4)

        with tf.variable_scope('Uconv1') as scope:

            with tf.variable_scope('UConv1_1') as scope:
                UConv1_1_bias_out =  conv3d_xavier(self.Fuse4_out,8)
                bn_out = batch_norm(UConv1_1_bias_out,self.is_batch_stats)
                self.UConv1_1_out = tf.nn.relu(bn_out, name='ReLU')

            with tf.variable_scope('UConv1_2') as scope:
                bias_out = conv3d_xavier(self.UConv1_1_out, 8)
                bn_out = batch_norm(bias_out, self.is_batch_stats)
                self.UConv1_2_out = tf.nn.relu(bn_out, name='ReLU')

            with tf.variable_scope('UConv1_3') as scope:
                print('Num of classes in the output is', num_classes)
                self.UConv1_3_out = conv3d_xavier(self.UConv1_2_out, num_classes,k_d=1,k_h=1,k_w=1)

        return self.UConv1_3_out

    def dice_loss(self, prob_per_class, correct_label, num_class):

        with tf.name_scope("DtypeChange") as scope:
            correct_label = tf.cast(correct_label, dtype=tf.float32)
            prob_per_class = tf.cast(prob_per_class, dtype=tf.float32)

        with tf.name_scope("Reshape") as scope:
            correct_label = tf.reshape(correct_label, [-1])
            r_labels = tf.one_hot(tf.cast(correct_label, dtype=tf.int64), depth=num_class, name='binarize')
            r_labels = tf.cast(r_labels, dtype=tf.float32)
            r_prob_per_class = tf.reshape(prob_per_class, [-1, num_class], name='reshape_labels')

        # Mask out unlabeled voxels.
        with tf.name_scope('gather') as scope:
            vessel_bool = tf.greater_equal(correct_label, 0)
            loss_samples = tf.where(vessel_bool)
            g_r_labels = tf.reshape(tf.gather(r_labels, loss_samples), [-1, num_class])
            g_r_prob_per_class = tf.reshape(tf.gather(r_prob_per_class, loss_samples), [-1, num_class])

        with tf.name_scope("DiceLoss"):
            intersection = tf.reduce_sum(tf.multiply(g_r_labels, g_r_prob_per_class), axis=0)
            denom1 = tf.reduce_sum(g_r_labels, axis=0)
            denom2 = tf.reduce_sum(g_r_prob_per_class, axis=0)
            self.m_dice_scores = tf.divide(tf.multiply(tf.constant(2.0), intersection),
                                           tf.add(denom1 + denom2, tf.constant(1e-6)))
            m_exist_cond = tf.where(tf.greater(denom1, tf.constant([5, 5, 5, 5, 5], dtype=tf.float32)))
            m_dice_score_exist_cond = tf.gather(self.m_dice_scores, m_exist_cond)
            self.m_dice_score = tf.reduce_mean(m_dice_score_exist_cond)

            dice_loss = tf.subtract(1.0, self.m_dice_score)

        return dice_loss

    def TrainOpAdamWithoutEstimator(self, loss):

        self.global_step = tf.Variable(1, name="global_step", trainable=False, dtype=tf.float32)

        with tf.name_scope("TRAIN_OP") as scope:
            update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS)  # Update ops to calculate moving avg. mean, variance batch normalization
            with tf.control_dependencies(update_ops):
                grads = tf.train.AdamOptimizer(self.base_lr).compute_gradients(loss, colocate_gradients_with_ops=True)
                train_op = tf.train.AdamOptimizer(self.base_lr).apply_gradients(grads, global_step=self.global_step)

        return train_op