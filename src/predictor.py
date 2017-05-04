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

    def load_and_align_inception_data(self, data_dir, image_size, margin = 0, gpu_memory_fraction=1.0):
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
            img = misc.imread(os.path.expanduser(image_paths[i]))
            if img.ndim == 2:
                img = facenet.to_rgb(img)
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
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list[i] = prewhitened
        images = np.stack(img_list)
        return images

    def load_training_data_and_labels(self, samples_dir, image_size):
        train_set = facenet.get_dataset(samples_dir)
        train_image_path_list, train_label_list = facenet.get_image_paths_and_labels(train_set)

        train_image_list = facenet.load_data(train_image_path_list, False, False, image_size, True)

        return train_image_list, train_label_list, train_set

    def setup_knn_prediction_op(self, samples_labels, session, k):
        num_labels = np.max(samples_labels)+1

        embeddings_placeholder = tf.placeholder(tf.float32, shape=[None, 128], name='knn_embeddings')
        train_embeddings_placeholder = tf.placeholder(tf.float32, shape=[None, 128], name='knn_train_embeddings')
        num_labels_placeholder = tf.placeholder(tf.int32, name='knn_num_labels')
        train_labels_placeholder = tf.placeholder(tf.int32, shape=[len(samples_labels)], name='knn_train_labels')

        train_labels_one_hot = tf.one_hot(train_labels_placeholder, num_labels_placeholder)

        # L2
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(train_embeddings_placeholder, tf.expand_dims(embeddings_placeholder,1))), axis=2))
        
        # Predict: Get min distance index (Nearest neighbor)
        top_k_samples_distances, top_k_samples_indices = tf.nn.top_k(tf.negative(distance), k=k)
        prediction_indices = tf.gather(train_labels_one_hot, top_k_samples_indices)

        # Predict the mode category
        count_of_predictions = tf.reduce_sum(prediction_indices, axis=1)
        prediction_op = tf.argmax(count_of_predictions, axis=1, name='knn_predictions')

        return prediction_op

    def load_frozen_model(self, model_path):
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="prefix", op_dict=None, producer_op_list=None)

    def load_facenet_model(self, model_dir, meta_file, ckpt_file):
        model_dir_exp = os.path.expanduser(model_dir)
        saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
        tf.get_default_session().run(tf.global_variables_initializer())
        tf.get_default_session().run(tf.local_variables_initializer())
        saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))

    def store_model(self, session, model_store_dir):
        model_dir = os.path.expanduser(model_store_dir)
        if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
            os.makedirs(model_dir)
        saver = tf.train.Saver()
        saver.save(session, model_dir + "/knn_classifier_model.ckpt")
        tf.train.write_graph(session.graph_def, '', model_dir + '/knn_classifier_model_graph.pb')

def main(args):
    # get file list    
    fp = FaceNetPredictor()
    model_dir = args.model_dir

    # load graph into session from checkpoint
    with tf.Graph().as_default():
        with tf.Session() as sess:

            log_dir = fp.createLogsDirIfNecessary(args.logs_base_dir)
            fp.storeRevisionInfoIfNecessary(log_dir)

            # load images
            train_images_list, train_labels_list, data_set = fp.load_training_data_and_labels(args.samples_dir, args.image_size)
            inception_images = fp.load_and_align_inception_data(args.data_dir, args.image_size)
            num_labels = np.max(train_labels_list)+1

            if args.frozen_model_path != None:
                print("Loading frozen model: %s" % args.frozen_model_path)
                fp.load_frozen_model(args.frozen_model_path)
            else:
                print('Model directory: %s' % model_dir)
                meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(model_dir))
                
                print('Metagraph file: %s' % meta_file)
                print('Checkpoint file: %s' % ckpt_file)
                fp.load_facenet_model(model_dir, meta_file, ckpt_file)
                fp.setup_knn_prediction_op(train_labels_list, sess, args.k)

            print("Successfully loaded model")
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings_placeholder = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            knn_embeddings_placeholder = tf.get_default_graph().get_tensor_by_name("knn_embeddings:0")
            knn_train_embeddings_placeholder = tf.get_default_graph().get_tensor_by_name("knn_train_embeddings:0")
            knn_num_labels_placeholder = tf.get_default_graph().get_tensor_by_name("knn_num_labels:0")
            knn_train_labels_placeholder = tf.get_default_graph().get_tensor_by_name("knn_train_labels:0")
            knn_predictions = tf.get_default_graph().get_tensor_by_name("knn_predictions:0")

            # calculate embedding for training images
            feed_dict = {images_placeholder: train_images_list, phase_train_placeholder: False}
            train_embeddings = sess.run(embeddings_placeholder, feed_dict=feed_dict)

            # calculate embedding for inception images
            feed_dict = {images_placeholder: inception_images, phase_train_placeholder: False}
            embeddings = sess.run(embeddings_placeholder, feed_dict=feed_dict)

            feed_dict = {knn_embeddings_placeholder:embeddings, knn_train_embeddings_placeholder: train_embeddings, knn_train_labels_placeholder: train_labels_list, knn_num_labels_placeholder: num_labels}
            predicted_classes = sess.run(knn_predictions, feed_dict=feed_dict)

            predicted_class_labels = map(lambda imageClass: imageClass.name, np.take(data_set, predicted_classes))
            print("Predictions")
            print(predicted_classes)
            print(predicted_class_labels)

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
        help='The parameter k for determining the k-nearest neighbors. Rule-of-thumb: should not be larger than the number of samples in each class.', default=2)

    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))