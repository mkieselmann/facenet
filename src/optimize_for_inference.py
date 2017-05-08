from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib
import tensorflow as tf 

FLAGS = None


def main(unused_args):
  if not gfile.Exists(FLAGS.input):
    print("Input graph file '" + FLAGS.input + "' does not exist!")
    return -1

  input_graph_def = graph_pb2.GraphDef()
  with gfile.Open(FLAGS.input, "rb") as f:
    data = f.read()
    if FLAGS.frozen_graph:
      input_graph_def.ParseFromString(data)
    else:
      text_format.Merge(data.decode("utf-8"), input_graph_def)

  enum = FLAGS.placeholder_type_enum.split(",")
  enum_int = map(int, enum)

  optimized_graph_def = replace_phase_train_with_constant(input_graph_def)

  output_graph_def = optimize_for_inference_lib.optimize_for_inference(
      optimized_graph_def,
      FLAGS.input_names.split(","),
      FLAGS.output_names.split(","), 
      enum_int)

  if FLAGS.frozen_graph:
    f = gfile.FastGFile(FLAGS.output, "w")
    f.write(output_graph_def.SerializeToString())
  else:
    graph_io.write_graph(output_graph_def,
                         os.path.dirname(FLAGS.output),
                         os.path.basename(FLAGS.output))
  return 0

def replace_phase_train_with_constant(input_graph_def):
  output_graph_def = graph_pb2.GraphDef()

  example_graph = tf.Graph()
  with tf.Session(graph=example_graph):
    c = tf.constant(False, dtype=bool, shape=[], name='phase_train')
    for node in example_graph.as_graph_def().node:
      if node.name == 'phase_train':
        phase_train_constant_def = node

  for input_node in input_graph_def.node:
    output_node = node_def_pb2.NodeDef()
    if input_node.name in "phase_train":
      output_node.CopyFrom(phase_train_constant_def)
    else:
      output_node.CopyFrom(input_node)
    output_graph_def.node.extend([output_node])

  return output_graph_def

def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--input",
      type=str,
      default="",
      help="TensorFlow \'GraphDef\' file to load.")
  parser.add_argument(
      "--output",
      type=str,
      default="",
      help="File to save the output graph to.")
  parser.add_argument(
      "--input_names",
      type=str,
      default="",
      help="Input node names, comma separated.")
  parser.add_argument(
      "--output_names",
      type=str,
      default="",
      help="Output node names, comma separated.")
  parser.add_argument(
      "--frozen_graph",
      nargs="?",
      const=True,
      type="bool",
      default=True,
      help="""\
      If true, the input graph is a binary frozen GraphDef
      file; if false, it is a text GraphDef proto file.\
      """)
  parser.add_argument(
      "--placeholder_type_enum",
      type=str,
      default=str(dtypes.float32.as_datatype_enum),
      help="The AttrValue enum to use for placeholders.")
  return parser.parse_known_args()


if __name__ == "__main__":
  FLAGS, unparsed = parse_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)