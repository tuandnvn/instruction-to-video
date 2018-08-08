from simple_model import SimpleAttentionModel

class MultiAttentionModel(SimpleAttentionModel):
	def _build_visual_encoder(self, hparams):
		"""
		Load a pretrained cnn network and run the image through 

		Return:

		visual_encoder_outputs: k_1 to k_H^2
		visual_encoder_state: cell state
		"""
		pass


	def _build_visual_decoder(self, encoder_outputs, encoder_state, visual_encoder_outputs, visual_encoder_state, hparams): 
		"""Build and run a RNN decoder with a final projection layer.

        Args:
          encoder_outputs: The outputs of encoder for every time step.
          encoder_state: The final state of the encoder.
          visual_encoder_outputs: The outputs of visual encoder for all visual regions
          visual_encoder_state

          hparams: The Hyperparameters configurations.

        Returns:
          A tuple of final logits and final decoder state:
            logits: size [time, batch_size, vocab_size] when time_major=True.
        """