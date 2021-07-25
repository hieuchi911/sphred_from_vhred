import tensorflow as tf
import tensorflow_addons as tfa
from configs import args

def create_multi_rnn_cell(rnn_type, hidden_dim, keep_prob, num_layer):
	def single_rnn_cell():
		if rnn_type.lower() == "lstm":
			cell = tf.keras.layers.LSTMCell(hidden_dim, dropout=keep_prob)
		elif rnn_type.lower() == "gru":
			cell = tf.keras.layers.GRUCell(hidden_dim, dropout=keep_prob)
		else:
			raise ValueError(" # Unsupported rnn_type: %s." % rnn_type)
		return cell
	# accepted and returned states are n-tuples where n=len(cells)
	cell = tf.keras.layers.StackedRNNCells([single_rnn_cell() for _ in range(num_layer)])
	return cell

def draw_z_prior(batch_size):
	return tf.random.normal([batch_size, args['latent_size']])

def reparamter_trick(z_mean, z_std):
	z = z_mean + z_std * draw_z_prior(tf.shape(z_mean)[0])
	return z

def kl_weights_fn(global_step):
	return args['anneal_max'] * tf.sigmoid((150 / args['anneal_bias']) * (
			tf.cast(global_step, dtype=tf.float32) - tf.constant(args['anneal_bias']/ 2)))

def kl_loss_fn(mean_1, std_1, mean_2, std_2):
	return 0.5 * tf.reduce_sum(input_tensor=tf.math.square(std_1) / tf.math.square(std_2) +
                             tf.math.square(mean_2 - mean_1) / tf.math.square(std_2) - 1 +
                             2 * tf.math.log(std_2) - 2 * tf.math.log(std_1)) / tf.cast(args['batch_size'], dtype=tf.float32)
