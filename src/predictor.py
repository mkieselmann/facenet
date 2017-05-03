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

def load_and_align_data(image_paths, image_size, margin = 0, gpu_memory_fraction=1.0):
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

    def load_training_data_and_labels(self, samples_dir, image_size):
        train_set = facenet.get_dataset(samples_dir)
        train_image_path_list, train_label_list = facenet.get_image_paths_and_labels(train_set)

        train_image_list = facenet.load_data(train_image_path_list, False, False, image_size, True)

        return train_image_list, train_label_list, train_set

    def knn(self, embeddings, samples_embedding_list, samples_labels_oneHot, session, k=4):
        embeddings_placeholder = tf.placeholder(tf.float32, shape=[None, 128])
        train_embeddings_placeholder = tf.placeholder(tf.float32, shape=[None, 128])
        train_labels_placeholder = tf.placeholder(tf.int32, shape=[None,samples_labels_oneHot.shape[1]])

        # L1
        #distance = tf.reduce_sum(tf.abs(tf.subtract(train_embeddings_placeholder, tf.expand_dims(embeddings_placeholder,1))), axis=2)

        # L2
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(train_embeddings_placeholder, tf.expand_dims(embeddings_placeholder,1))), axis=2))
        
        # Predict: Get min distance index (Nearest neighbor)
        top_k_samples_distances, top_k_samples_indices = tf.nn.top_k(tf.negative(distance), k=k)
        prediction_indices = tf.gather(train_labels_placeholder, top_k_samples_indices)

        # Predict the mode category
        count_of_predictions = tf.reduce_sum(prediction_indices, axis=1)
        prediction_op = tf.argmax(count_of_predictions, axis=1)

        feed_dict = {embeddings_placeholder:embeddings, train_embeddings_placeholder: samples_embedding_list, train_labels_placeholder: samples_labels_oneHot}
        predictions = session.run(prediction_op, feed_dict=feed_dict)

        return predictions

def main(args):
    # get file list    
    fp = FaceNetPredictor()
    files = glob.glob(args.data_dir + '/*')
    images = load_and_align_data(files, args.image_size)
    model_dir = args.model_dir

    # load graph into session from checkpoint
    with tf.Graph().as_default():
        with tf.Session() as sess:

            log_dir = fp.createLogsDirIfNecessary(args.logs_base_dir)
            fp.storeRevisionInfoIfNecessary(log_dir)

            print('Model directory: %s' % model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(model_dir))
                
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(model_dir, meta_file, ckpt_file)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings_placeholder = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            feed_dict = {images_placeholder:images, phase_train_placeholder:False}
            embeddings = sess.run(embeddings_placeholder,feed_dict=feed_dict)

            train_images_list, train_labels_list, data_set = fp.load_training_data_and_labels(args.samples_dir, args.image_size)
            feed_dict = {images_placeholder:train_images_list, phase_train_placeholder:False}
            train_embeddings = sess.run(embeddings_placeholder,feed_dict=feed_dict)

            num_labels = np.max(train_labels_list)+1

            num_labels_placeholder = tf.placeholder(tf.int32)
            train_labels_placeholder = tf.placeholder(tf.int32, shape=[len(train_labels_list)])
            one_hot_op = tf.one_hot(train_labels_placeholder, num_labels_placeholder)

            feed_dict = {num_labels_placeholder: num_labels, train_labels_placeholder: train_labels_list}
            train_labels_list_one_hot = sess.run(one_hot_op, feed_dict=feed_dict)

            k = 2
            predictions = fp.knn(embeddings, train_embeddings, train_labels_list_one_hot, sess, k)
            print("Predictions")
            print(predictions)

            predicted_labels = []
            for prediction in predictions:
                class_name = data_set[prediction].name
                predicted_labels.append(class_name)
            print(predicted_labels)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, required=False,
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--model_dir', type=str, required=True,
        help='Directory where to load the model from.')
    parser.add_argument('--data_dir', type=str, required=True,
        help='Path to the data directory containing images for which to create predictions.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--samples_dir', type=str, required=True,
        help='Path to the directory containing folders for each class with training images.')

    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))