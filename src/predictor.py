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
    
    # convert images to their tensor representation
    def convert(self, image_file, image_size):
        current = Image.open(image_file)
        image_size = image_size
        file_contents = tf.read_file(image_file)
        name = image_file.rsplit('/')[-2]
        image = tf.image.decode_png(file_contents)#, channels=3)
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image.set_shape((image_size, image_size, 3))
        #image = tf.image.per_image_whitening(image)
        image = tf.expand_dims(image, 0, name = name)
        return image

def main(args):
    # get file list    
    files = glob.glob(args.data_dir + '/*')
    model_dir = args.model_dir

    # load graph into session from checkpoint
    with tf.Graph().as_default():
        with tf.Session() as sess:
            fp = FaceNetPredictor()

            log_dir = fp.createLogsDirIfNecessary(args.logs_base_dir)
            fp.storeRevisionInfoIfNecessary(log_dir)

            print('Model directory: %s' % model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(model_dir))
                
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(model_dir, meta_file, ckpt_file)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            softmax_placeholder = tf.get_default_graph().get_tensor_by_name("softmax:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            for x, file in enumerate(files):
                print("Predicting for image: %s" % file)
                image = fp.convert(file, args.image_size)
                feed_dict = {images_placeholder:image.eval(), phase_train_placeholder:False}
                softmax = sess.run(softmax_placeholder,feed_dict=feed_dict)
                softmax_out = softmax[0].argmax()
                print("full vector: {}, max: {}", softmax, softmax_out)

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

    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))