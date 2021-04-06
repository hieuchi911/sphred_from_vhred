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
	return tf.random.truncated_normal([args['batch_size'], args['latent_size']])

def reparamter_trick(z_mean, z_logvar):
	z = z_mean + tf.exp(0.5 * z_logvar) * draw_z_prior()
	return z

def kl_weights_fn(global_step):
	return args['anneal_max'] * tf.sigmoid((10 / args['anneal_bias']) * (
			tf.cast(global_step, dtype=tf.float32) - tf.constant(2 * args['anneal_bias']/ 2)))

def kl_loss_fn(mean_1, log_var_1, mean_2, log_var_2):
	return 0.5 * tf.reduce_sum(input_tensor=log_var_1 / log_var_2 +
	                           (mean_2 - mean_1) / log_var_2 * (mean_2 - mean_1) - 1 +
	                           tf.math.log(log_var_2) - tf.math.log(log_var_1)) / tf.cast(args['batch_size'], dtype=tf.float32)

# def nucleus_sampling():
#     assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
#     top_k = min(top_k, logits.size(-1))  # Safety check
#     if top_k > 0:
#         # Remove all tokens with a probability less than the last token of the top-k
#         indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
#         logits[indices_to_remove] = filter_value

#     if top_p > 0.0:
#         sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#         cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

#         # Remove tokens with cumulative probability above the threshold
#         sorted_indices_to_remove = cumulative_probs > top_p
#         # Shift the indices to the right to keep also the first token above the threshold
#         sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#         sorted_indices_to_remove[..., 0] = 0

#         indices_to_remove = sorted_indices[sorted_indices_to_remove]
#         logits[indices_to_remove] = filter_value
#     return logits

# # Here is how to use this function for top-p sampling
# temperature = 1.0
# top_k = 0
# top_p = 0.9

# # Get logits with a forward pass in our model (input is pre-defined)
# logits = model(input)

# # Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
# logits = logits[0, -1, :] / temperature
# filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

# # Sample from the filtered distribution
# probabilities = F.softmax(filtered_logits, dim=-1)
# next_token = torch.multinomial(probabilities, 1)