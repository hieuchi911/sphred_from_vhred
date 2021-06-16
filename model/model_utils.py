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
		# Apply dropout mechanism to the inputs and outputs by multiplying with the corresponding probabilities
		# cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
		return cell
	# accepted and returned states are n-tuples where n=len(cells)
	cell = tf.keras.layers.StackedRNNCells([single_rnn_cell() for _ in range(num_layer)])
	# cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(num_layer)], state_is_tuple=False)
	return cell

def encoder(embedded_inputs, lengths, hparams, keep_prob):
	rnn_type = hparams.rnn_type
	hidden_dim = hparams.hidden_dim
	num_layer = hparams.num_layer

	encoder_cell = create_multi_rnn_cell(rnn_type, hidden_dim, keep_prob, num_layer)
	outputs, states = tf.compat.v1.nn.dynamic_rnn(
		cell=encoder_cell, inputs=embedded_inputs, sequence_length=lengths, dtype=tf.float32)

	return outputs, states
	# outputs: [batch_size, enc_max_len, hidden_dim]
	# states: ([batch_size, hidden_dim]) * num_layer

def draw_z_prior():
	return tf.random.normal([args['batch_size'], args['latent_size']])

def reparamter_trick(z_mean, z_std):
	z = z_mean + z_std * draw_z_prior()
	return z

def kl_weights_fn(global_step):
	return args['anneal_max'] * tf.sigmoid((150 / args['anneal_bias']) * (
			tf.cast(global_step, dtype=tf.float32) - tf.constant(args['anneal_bias']/ 2)))

def kl_loss_fn(mean_1, std_1, mean_2, std_2):
	return 0.5 * tf.reduce_sum(input_tensor=tf.math.square(std_1) / tf.math.square(std_2) +
	                           tf.math.square(mean_2 - mean_1) / tf.math.square(std_2) - 1 +
	                           2 * tf.math.log(std_2) - 2 * tf.math.log(std_1)) / tf.cast(args['batch_size'], dtype=tf.float32)
