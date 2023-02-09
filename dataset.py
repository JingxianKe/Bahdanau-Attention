# import the necessary packages
import tensorflow_text as tf_text
import tensorflow as tf
import random

# define a module level autotune
_AUTO = tf.data.AUTOTUNE


def load_data(fname):
	# open the file with utf-8 encoding
	with open(fname, "r", encoding="utf-8") as textFile:
		# the source and the target sentence is demarcated with tab
		# iterate over each line and split the sentences to get
		# the individual source and target sentence pairs
		lines = textFile.readlines()
		pairs = [line.split("\t")[:] for line in lines]

		# randomly shuffle the pairs
		random.shuffle(pairs)

		# collect the source sentences and target sentences into
		# respective lists
		source = [src for src, _ in pairs]
		target = [trgt for _, trgt in pairs]
	# return the list of source and target sentences
	return (source, target)

def splitting_dataset(source, target):
		# calculate the training and validation size
		trainSize = int(len(source) * 0.8)
		valSize = int(len(source) * 0.1)
		# split the inputs into train, val, and test
		(trainSource, trainTarget) = (source[: trainSize],
				target[: trainSize])
		(valSource, valTarget) = (source[trainSize : trainSize + valSize],
				target[trainSize : trainSize + valSize])
		(testSource, testTarget) = (source[trainSize + valSize :],
				target[trainSize + valSize :])
		# return the splits
		return (
				(trainSource, trainTarget),
				(valSource, valTarget),
				(testSource, testTarget),
		)

def make_dataset(splits, batchSize, train=False):
		# build a TensorFlow dataset from the input and target
		(source, target) = splits
		dataset = tf.data.Dataset.from_tensor_slices((source, target))
		# check if this is the training dataset, if so, shuffle, batch,
		# and prefetch it
		if train:
				dataset = (
						dataset
						.shuffle(dataset.cardinality().numpy())
						.batch(batchSize)
						.prefetch(_AUTO)
				)
		# otherwise, just batch the dataset
		else:
				dataset = (
						dataset
						.batch(batchSize)
						.prefetch(_AUTO)
				)
		# return the dataset
		return dataset

def tf_lower_and_split_punct(text):
		# split accented characters
		text = tf_text.normalize_utf8(text, "NFKD")
		text = tf.strings.lower(text)
		# keep space, a to z, and selected punctuations
		text = tf.strings.regex_replace(text, "[^ a-z.?!,]", "")
		# add spaces around punctuation
		text = tf.strings.regex_replace(text, "[.?!,]", r" \0 ")
		# strip whitespace and add [START] and [END] tokens
		text = tf.strings.strip(text)
		text = tf.strings.join(["[START]", text, "[END]"], separator=" ")
		# return the processed text
		return text







