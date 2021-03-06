import tensorflow as tf 
import os
import tensorflow.contrib.slim as slim 
import importlib
import random
from PIL import Image
import glob
import facenet
import sys
import argparse
from datetime import datetime
import align.detect_face
from scipy import misc
import numpy as np

class FaceNetPredictor:
    def createLogsDirIfNecessary(self, base_dir):
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        log_dir = os.path.join(os.path.expanduser(base_dir), subdir)
        if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(log_dir)

        return log_dir

    def storeRevisionInfoIfNecessary(self, log_dir):
        src_path,_ = os.path.split(os.path.realpath(__file__))
        facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    def load_and_align_inception_data(self, data_dir, session, image_size, margin = 0, gpu_memory_fraction=1.0):
        image_paths = glob.glob(data_dir + '/*')

        minsize = 20 # minimum size of face
        threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        factor = 0.709 # scale factor
        
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

        nrof_samples = len(image_paths)
        img_list = [None] * nrof_samples
        for i in xrange(nrof_samples):
            feed_dict = {"preprocess/image_path:0": os.path.expanduser(image_paths[i])}
            img = session.run("preprocess/image_array:0", feed_dict=feed_dict)

            img = img[:,:,0:3]
            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            det = np.squeeze(bounding_boxes[0,0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            #aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            #prewhitened = facenet.prewhiten(aligned)
            feed_dict = {"preprocess/image_array:0": cropped}
            processed_image = session.run("preprocess/pre_process_image:0", feed_dict=feed_dict)
            img_list[i] = processed_image
        images = np.stack(img_list)
        return images

    def setup_load_and_pre_process_image_op(self, image_size):
        image_path_placeholder = tf.placeholder(tf.string, name='preprocess/image_path')
        image_contents_placeholder = tf.read_file(image_path_placeholder, name='preprocess/image_contents')   
        image = tf.image.decode_image(image_contents_placeholder)
        image = tf.identity(image, name="preprocess/image_array")
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image.set_shape((image_size, image_size, 3))
        image = tf.image.per_image_standardization(image)
        return tf.identity(image, name="preprocess/pre_process_image")

    def load_training_data(self, train_image_path_list, sess):
        train_image_list = []
        for path in train_image_path_list:
            feed_dict = {"preprocess/image_path:0": os.path.expanduser(path)}
            processed_image = sess.run("preprocess/pre_process_image:0", feed_dict=feed_dict)
            train_image_list.append(processed_image)

        #train_image_list = facenet.load_data(train_image_path_list, False, False, image_size, True)
        return train_image_list

    def get_image_paths_and_labels(self, samples_dir):
        train_set = facenet.get_dataset(samples_dir)
        train_image_path_list, train_labels_list = facenet.get_image_paths_and_labels(train_set)
        labels_list = map(lambda imageClass: imageClass.name, train_set)

        return train_image_path_list, train_labels_list, labels_list

    def setup_knn_prediction_op(self, samples_labels, session, k):
        num_labels = np.max(samples_labels)+1

        embeddings_placeholder = tf.placeholder(tf.float32, shape=[None, 128], name='knn_embeddings')
        train_embeddings_placeholder = tf.placeholder(tf.float32, shape=[None, 128], name='knn_train_embeddings')
        num_labels_placeholder = tf.placeholder(tf.int32, name='knn_num_labels')
        train_labels_placeholder = tf.placeholder(tf.int32, shape=[None], name='knn_train_labels')#len(samples_labels)

        train_labels_one_hot = tf.one_hot(train_labels_placeholder, num_labels_placeholder)

        # L2
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(train_embeddings_placeholder, tf.expand_dims(embeddings_placeholder,1))), axis=2))
        
        # Predict: Get min distance index (Nearest neighbor)
        top_k_samples_negative_distances, top_k_samples_indices = tf.nn.top_k(tf.negative(distance), k=k)
        prediction_indices = tf.gather(train_labels_one_hot, top_k_samples_indices)
        top_k_samples_distances = tf.negative(top_k_samples_negative_distances)

        distance_weights = tf.expand_dims(tf.div(tf.ones([1], tf.float32), top_k_samples_distances), 1)
        weighted_count_of_predictions = tf.squeeze(tf.matmul(distance_weights, prediction_indices), axis=[1])
        prediction_op = tf.argmax(weighted_count_of_predictions, axis=1, name='knn_predictions')

        # Predict the mode category
        #count_of_predictions = tf.reduce_sum(prediction_indices, axis=1)
        #prediction_op = tf.argmax(count_of_predictions, axis=1, name='knn_predictions')

        # get distance for prediction
        # predictions [40,40,40,40] -> [[40,40,40],[40,40,40],[40,40,40],[40,40,40]]: #predictions x k
        top_k_labels = tf.gather(train_labels_placeholder, top_k_samples_indices)
        predictions_repeated = tf.matmul(tf.cast(tf.expand_dims(prediction_op, 1), tf.float32), tf.ones([1,k]))
        distances_for_predicted_classes = tf.where(tf.equal(tf.cast(top_k_labels, tf.float32), predictions_repeated), top_k_samples_distances, tf.add(top_k_samples_distances, 1000.0))
        min_distances = tf.reduce_min(distances_for_predicted_classes, axis=1, name='knn_min_distances')

        return prediction_op, min_distances

    def load_facenet_model(self, model_dir, meta_file, ckpt_file):
        model_dir_exp = os.path.expanduser(model_dir)
        saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
        tf.get_default_session().run(tf.global_variables_initializer())
        tf.get_default_session().run(tf.local_variables_initializer())
        saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))

    def load_frozen_model(self, model_path):
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        return graph

    def store_model(self, session, model_store_dir):
        model_dir = os.path.expanduser(model_store_dir)
        datetime_str = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % datetime_str)
        metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % datetime_str)
        if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
            os.makedirs(model_dir)
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        saver.save(session, checkpoint_path, write_meta_graph=False, global_step=1)
        saver.export_meta_graph(metagraph_filename)

def main(args):
    # get file list    
    fp = FaceNetPredictor()
    model_dir = args.model_dir

    # load graph into session from checkpoint
    with tf.Graph().as_default():
        with tf.Session() as sess:

            log_dir = fp.createLogsDirIfNecessary(args.logs_base_dir)
            fp.storeRevisionInfoIfNecessary(log_dir)

            train_image_path_list, train_labels_list, labels_list = fp.get_image_paths_and_labels(args.samples_dir)

            if args.frozen_model_path != None:
                print("Loading frozen model: %s" % args.frozen_model_path)
                graph = fp.load_frozen_model(args.frozen_model_path)
                sess = tf.Session(graph=graph)
                sess.as_default()
            else:
                print('Model directory: %s' % model_dir)
                meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(model_dir))
                
                print('Metagraph file: %s' % meta_file)
                print('Checkpoint file: %s' % ckpt_file)
                fp.setup_load_and_pre_process_image_op(args.image_size)
                fp.load_facenet_model(model_dir, meta_file, ckpt_file)
                fp.setup_knn_prediction_op(train_labels_list, sess, args.k)
            print("Successfully loaded model")

            train_images = fp.load_training_data(train_image_path_list, sess)
            print("Successfully loaded training images")
            inception_images = fp.load_and_align_inception_data(args.data_dir, sess, args.image_size)
            print("Successfully loaded inception images")

            # calculate embedding for training images
            feed_dict = {"input:0": train_images, "phase_train:0": False}
            train_embeddings = sess.run("embeddings:0", feed_dict=feed_dict)
            print("Calculated training images embeddings")

            # calculate embedding for inception images
            feed_dict = {"input:0": inception_images, "phase_train:0": False}
            embeddings = sess.run("embeddings:0", feed_dict=feed_dict)
            print("Calculated inception images embeddings")

            feed_dict = {"knn_embeddings:0":embeddings, "knn_train_embeddings:0": train_embeddings, "knn_train_labels:0": train_labels_list, "knn_num_labels:0": len(labels_list)}
            predicted_classes, distances = sess.run(["knn_predictions:0",'knn_min_distances:0'], feed_dict=feed_dict)

            print("Predictions")
            print(predicted_classes)

            predicted_class_labels = np.take(labels_list, predicted_classes)
            print(predicted_class_labels)

            print("Distances")
            print(distances)

            if args.model_store_dir != None:
                fp.store_model(sess, args.model_store_dir)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, required=False,
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--model_dir', type=str, required=True,
        help='Directory where to load the facenet model from.')
    parser.add_argument('--frozen_model_path', type=str, required=False,
        help='Directory where to load the complete frozen model from.', default=None)
    parser.add_argument('--model_store_dir', type=str, required=False,
        help='Directory where to store the model subdirectory.', default=None)
    parser.add_argument('--data_dir', type=str, required=True,
        help='Path to the data directory containing images for which to create predictions.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--samples_dir', type=str, required=True,
        help='Path to the directory containing folders for each class with training images.')
    parser.add_argument('--k', type=int, required=False,
        help='The parameter k for determining the k-nearest neighbors. Rule-of-thumb: should not be larger than the number of samples in each class.', default=3)

    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))