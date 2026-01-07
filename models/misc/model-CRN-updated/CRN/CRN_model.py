# Copyright (c) 2020, Ioana Bica
# Converted to PyTorch 2025

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils.flip_gradient import flip_gradient
import numpy as np
import os
import logging


class CRN_Model(nn.Module):
    def __init__(self, params, hyperparams, b_train_decoder=False):
        super(CRN_Model, self).__init__()

        self.num_treatments = params['num_treatments']
        self.num_covariates = params['num_covariates']
        self.num_outputs = params['num_outputs']
        self.max_sequence_length = params['max_sequence_length']
        self.num_epochs = params['num_epochs']

        self.br_size = hyperparams['br_size']
        self.rnn_hidden_units = hyperparams['rnn_hidden_units']
        self.fc_hidden_units = hyperparams['fc_hidden_units']
        self.batch_size = hyperparams['batch_size']
        self.rnn_keep_prob = hyperparams['rnn_keep_prob']
        self.learning_rate = hyperparams['learning_rate']

        self.b_train_decoder = b_train_decoder

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build network layers
        self._build_network()

    def _build_network(self):
        # RNN for building balanced representation
        self.rnn = nn.LSTM(
            input_size=self.num_covariates + self.num_treatments,
            hidden_size=self.rnn_hidden_units,
            batch_first=True,
            dropout=1.0 - self.rnn_keep_prob if self.rnn_keep_prob < 1.0 else 0.0
        )

        # Balancing representation layer
        self.br_layer = nn.Sequential(
            nn.Linear(self.rnn_hidden_units, self.br_size),
            nn.ELU()
        )

        # Treatment prediction network (with gradient reversal)
        self.treatment_network = nn.Sequential(
            nn.Linear(self.br_size, self.fc_hidden_units),
            nn.ELU(),
            nn.Linear(self.fc_hidden_units, self.num_treatments)
        )

        # Outcome prediction network
        self.outcome_network = nn.Sequential(
            nn.Linear(self.br_size + self.num_treatments, self.fc_hidden_units),
            nn.ELU(),
            nn.Linear(self.fc_hidden_units, self.num_outputs)
        )

    def compute_sequence_length(self, sequence):
        """Compute actual sequence lengths from padded sequences."""
        used = torch.sign(torch.max(torch.abs(sequence), dim=2)[0])
        length = torch.sum(used, dim=1)
        return length.long()

    def build_balancing_representation(self, current_covariates, previous_treatments, init_state=None):
        """Build treatment-invariant balanced representations."""
        rnn_input = torch.cat([current_covariates, previous_treatments], dim=-1)
        sequence_lengths = self.compute_sequence_length(rnn_input)

        # Initialize hidden state if needed (for decoder)
        if init_state is not None and self.b_train_decoder:
            h0 = init_state.unsqueeze(0)
            c0 = init_state.unsqueeze(0)
            initial_state = (h0, c0)
        else:
            initial_state = None

        # Apply dropout during training
        self.rnn.train(self.training)

        # Run through LSTM
        if initial_state is not None:
            rnn_output, _ = self.rnn(rnn_input, initial_state)
        else:
            rnn_output, _ = self.rnn(rnn_input)

        # Flatten to apply same weights to all time steps
        batch_size = rnn_output.shape[0]
        rnn_output_flat = rnn_output.reshape(-1, self.rnn_hidden_units)

        # Build balancing representation
        balancing_representation = self.br_layer(rnn_output_flat)

        return balancing_representation

    def build_treatment_predictions(self, balancing_representation, alpha):
        """Build treatment predictions with gradient reversal."""
        # Apply gradient reversal
        balancing_representation_gr = flip_gradient(balancing_representation, alpha)

        # Predict treatments
        treatment_logits = self.treatment_network(balancing_representation_gr)
        treatment_probs = torch.softmax(treatment_logits, dim=-1)

        return treatment_probs

    def build_outcome_predictions(self, balancing_representation, current_treatments):
        """Build outcome predictions."""
        batch_size = current_treatments.shape[0]
        current_treatments_flat = current_treatments.reshape(-1, self.num_treatments)

        # Concatenate balanced representation with treatments
        outcome_input = torch.cat([balancing_representation, current_treatments_flat], dim=-1)

        # Predict outcomes
        outcome_predictions = self.outcome_network(outcome_input)

        return outcome_predictions

    def compute_loss_treatments(self, target_treatments, treatment_predictions, active_entries):
        """Compute cross-entropy loss for treatment predictions."""
        treatment_predictions = treatment_predictions.reshape(-1, self.max_sequence_length, self.num_treatments)
        target_treatments_flat = target_treatments.reshape(-1, self.max_sequence_length, self.num_treatments)
        active_entries_flat = active_entries.reshape(-1, self.max_sequence_length, self.num_outputs)

        # Cross entropy loss
        loss = -torch.sum(target_treatments_flat * torch.log(treatment_predictions + 1e-8) * active_entries_flat) / torch.sum(active_entries_flat)

        return loss

    def compute_loss_outcomes(self, target_outputs, outcome_predictions, active_entries):
        """Compute MSE loss for outcome predictions."""
        outcome_predictions = outcome_predictions.reshape(-1, self.max_sequence_length, self.num_outputs)

        # MSE loss
        loss = torch.sum((target_outputs - outcome_predictions) ** 2 * active_entries) / torch.sum(active_entries)

        return loss

    def forward(self, current_covariates, previous_treatments, current_treatments, init_state=None, alpha=1.0):
        """Forward pass through the model."""
        # Build balanced representation
        balancing_representation = self.build_balancing_representation(
            current_covariates, previous_treatments, init_state
        )

        # Predict treatments (with gradient reversal)
        treatment_predictions = self.build_treatment_predictions(balancing_representation, alpha)

        # Predict outcomes
        outcome_predictions = self.build_outcome_predictions(balancing_representation, current_treatments)

        return balancing_representation, treatment_predictions, outcome_predictions

    def train_model(self, dataset_train, dataset_val, model_name, model_folder):
        """Train the CRN model."""
        self.to(self.device)

        # Setup optimizer
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            super(CRN_Model, self).train()  # Set model to training mode

            # Compute alpha for gradient reversal (schedule)
            p = float(epoch) / float(self.num_epochs)
            alpha_current = 2. / (1. + np.exp(-10. * p)) - 1

            iteration = 0
            for batch_data in self.gen_epoch(dataset_train, batch_size=self.batch_size):
                batch_current_covariates, batch_previous_treatments, batch_current_treatments, \
                batch_init_state, batch_outputs, batch_active_entries = batch_data

                # Convert to tensors
                batch_current_covariates = torch.FloatTensor(batch_current_covariates).to(self.device)
                batch_previous_treatments = torch.FloatTensor(batch_previous_treatments).to(self.device)
                batch_current_treatments = torch.FloatTensor(batch_current_treatments).to(self.device)
                batch_outputs = torch.FloatTensor(batch_outputs).to(self.device)
                batch_active_entries = torch.FloatTensor(batch_active_entries).to(self.device)

                if batch_init_state is not None:
                    batch_init_state = torch.FloatTensor(batch_init_state).to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                balancing_representation, treatment_predictions, outcome_predictions = self.forward(
                    batch_current_covariates, batch_previous_treatments, batch_current_treatments,
                    batch_init_state, alpha_current
                )

                # Compute losses
                loss_treatments = self.compute_loss_treatments(
                    batch_current_treatments, treatment_predictions, batch_active_entries
                )
                loss_outcomes = self.compute_loss_outcomes(
                    batch_outputs, outcome_predictions, batch_active_entries
                )
                total_loss = loss_outcomes + loss_treatments

                # Backward pass
                total_loss.backward()
                optimizer.step()

                iteration += 1

            logging.info(
                f"Epoch {epoch + 1} out of {self.num_epochs} | total loss = {total_loss.item():.4f} | "
                f"outcome loss = {loss_outcomes.item():.4f} | treatment loss = {loss_treatments.item():.4f} | "
                f"current alpha = {alpha_current:.4f}"
            )

        # Validation
        validation_loss, validation_loss_outcomes, validation_loss_treatments = self.compute_validation_loss(dataset_val)
        validation_mse, _ = self.evaluate_predictions(dataset_val)

        logging.info(
            f"Epoch {epoch} Summary| Validation total loss = {validation_loss:.4f} | "
            f"Validation outcome loss = {validation_loss_outcomes:.4f} | "
            f"Validation treatment loss {validation_loss_treatments:.4f} | "
            f"Validation mse = {validation_mse:.4f}"
        )

        # Save model
        checkpoint_name = model_name + "_final"
        self.save_network(model_folder, checkpoint_name)

    def load_model(self, model_name, model_folder):
        """Load a trained model."""
        checkpoint_name = model_name + "_final"
        self.load_network(model_folder, checkpoint_name)
        self.to(self.device)

    def gen_epoch(self, dataset, batch_size, training_mode=True):
        """Generate batches for an epoch."""
        dataset_size = dataset['current_covariates'].shape[0]
        num_batches = int(dataset_size / batch_size) + 1

        for i in range(num_batches):
            if i == num_batches - 1:
                batch_samples = range(dataset_size - batch_size, dataset_size)
            else:
                batch_samples = range(i * batch_size, (i + 1) * batch_size)

            batch_current_covariates = dataset['current_covariates'][batch_samples, :, :]
            batch_previous_treatments = dataset['previous_treatments'][batch_samples, :, :]
            batch_current_treatments = dataset['current_treatments'][batch_samples, :, :]

            # Handle previous treatments (add zero initial treatment)
            if not self.b_train_decoder:
                batch_size_actual = batch_previous_treatments.shape[0]
                zero_init_treatment = np.zeros(shape=[batch_size_actual, 1, self.num_treatments])
                batch_previous_treatments = np.concatenate([zero_init_treatment, batch_previous_treatments], axis=1)

            if training_mode:
                batch_outputs = dataset['outputs'][batch_samples, :, :]
                batch_active_entries = dataset['active_entries'][batch_samples, :, :]

                batch_init_state = None
                if self.b_train_decoder:
                    batch_init_state = dataset['init_state'][batch_samples, :]

                yield (batch_current_covariates, batch_previous_treatments, batch_current_treatments,
                       batch_init_state, batch_outputs, batch_active_entries)
            else:
                batch_init_state = None
                if self.b_train_decoder:
                    batch_init_state = dataset['init_state'][batch_samples, :]

                yield (batch_current_covariates, batch_previous_treatments, batch_current_treatments,
                       batch_init_state)

    def compute_validation_loss(self, dataset):
        """Compute validation loss."""
        self.eval()

        validation_losses = []
        validation_losses_outcomes = []
        validation_losses_treatments = []

        dataset_size = dataset['current_covariates'].shape[0]
        batch_size = 10000 if dataset_size > 10000 else dataset_size

        with torch.no_grad():
            for batch_data in self.gen_epoch(dataset, batch_size=batch_size):
                batch_current_covariates, batch_previous_treatments, batch_current_treatments, \
                batch_init_state, batch_outputs, batch_active_entries = batch_data

                # Convert to tensors
                batch_current_covariates = torch.FloatTensor(batch_current_covariates).to(self.device)
                batch_previous_treatments = torch.FloatTensor(batch_previous_treatments).to(self.device)
                batch_current_treatments = torch.FloatTensor(batch_current_treatments).to(self.device)
                batch_outputs = torch.FloatTensor(batch_outputs).to(self.device)
                batch_active_entries = torch.FloatTensor(batch_active_entries).to(self.device)

                if batch_init_state is not None:
                    batch_init_state = torch.FloatTensor(batch_init_state).to(self.device)

                # Forward pass
                balancing_representation, treatment_predictions, outcome_predictions = self.forward(
                    batch_current_covariates, batch_previous_treatments, batch_current_treatments,
                    batch_init_state, alpha=1.0
                )

                # Compute losses
                loss_treatments = self.compute_loss_treatments(
                    batch_current_treatments, treatment_predictions, batch_active_entries
                )
                loss_outcomes = self.compute_loss_outcomes(
                    batch_outputs, outcome_predictions, batch_active_entries
                )
                total_loss = loss_outcomes + loss_treatments

                validation_losses.append(total_loss.item())
                validation_losses_outcomes.append(loss_outcomes.item())
                validation_losses_treatments.append(loss_treatments.item())

        return np.mean(validation_losses), np.mean(validation_losses_outcomes), np.mean(validation_losses_treatments)

    def get_balancing_reps(self, dataset):
        """Get balancing representations for a dataset."""
        logging.info("Computing balancing representations.")
        self.eval()

        dataset_size = dataset['current_covariates'].shape[0]
        balancing_reps = np.zeros(shape=(dataset_size, self.max_sequence_length, self.br_size))

        batch_size = 10000 if dataset_size > 10000 else dataset_size
        num_batches = int(dataset_size / batch_size) + 1

        batch_id = 0
        num_samples = 50

        with torch.no_grad():
            for batch_data in self.gen_epoch(dataset, batch_size=batch_size, training_mode=False):
                batch_current_covariates, batch_previous_treatments, batch_current_treatments, batch_init_state = batch_data

                # Convert to tensors
                batch_current_covariates = torch.FloatTensor(batch_current_covariates).to(self.device)
                batch_previous_treatments = torch.FloatTensor(batch_previous_treatments).to(self.device)
                batch_current_treatments = torch.FloatTensor(batch_current_treatments).to(self.device)

                if batch_init_state is not None:
                    batch_init_state = torch.FloatTensor(batch_init_state).to(self.device)

                # Dropout samples (Monte Carlo dropout)
                total_predictions = np.zeros(shape=(batch_size, self.max_sequence_length, self.br_size))

                for sample in range(num_samples):
                    super(CRN_Model, self).train()  # Enable dropout
                    balancing_representation = self.build_balancing_representation(
                        batch_current_covariates, batch_previous_treatments, batch_init_state
                    )
                    br_outputs = balancing_representation.cpu().numpy()
                    br_outputs = br_outputs.reshape(-1, self.max_sequence_length, self.br_size)
                    total_predictions += br_outputs

                total_predictions /= num_samples

                if batch_id == num_batches - 1:
                    batch_samples = range(dataset_size - batch_size, dataset_size)
                else:
                    batch_samples = range(batch_id * batch_size, (batch_id + 1) * batch_size)

                batch_id += 1
                balancing_reps[batch_samples] = total_predictions

        self.eval()
        return balancing_reps

    def get_predictions(self, dataset):
        """Get outcome predictions for a dataset."""
        logging.info("Performing one-step-ahead prediction.")
        self.eval()

        dataset_size = dataset['current_covariates'].shape[0]
        predictions = np.zeros(shape=(dataset_size, self.max_sequence_length, self.num_outputs))

        batch_size = 10000 if dataset_size > 10000 else dataset_size
        num_batches = int(dataset_size / batch_size) + 1

        batch_id = 0
        num_samples = 50

        with torch.no_grad():
            for batch_data in self.gen_epoch(dataset, batch_size=batch_size, training_mode=False):
                batch_current_covariates, batch_previous_treatments, batch_current_treatments, batch_init_state = batch_data

                # Convert to tensors
                batch_current_covariates = torch.FloatTensor(batch_current_covariates).to(self.device)
                batch_previous_treatments = torch.FloatTensor(batch_previous_treatments).to(self.device)
                batch_current_treatments = torch.FloatTensor(batch_current_treatments).to(self.device)

                if batch_init_state is not None:
                    batch_init_state = torch.FloatTensor(batch_init_state).to(self.device)

                # Dropout samples (Monte Carlo dropout)
                total_predictions = np.zeros(shape=(batch_size, self.max_sequence_length, self.num_outputs))

                for sample in range(num_samples):
                    super(CRN_Model, self).train()  # Enable dropout
                    _, _, outcome_predictions = self.forward(
                        batch_current_covariates, batch_previous_treatments, batch_current_treatments,
                        batch_init_state, alpha=1.0
                    )
                    predicted_outputs = outcome_predictions.cpu().numpy()
                    predicted_outputs = predicted_outputs.reshape(-1, self.max_sequence_length, self.num_outputs)
                    total_predictions += predicted_outputs

                total_predictions /= num_samples

                if batch_id == num_batches - 1:
                    batch_samples = range(dataset_size - batch_size, dataset_size)
                else:
                    batch_samples = range(batch_id * batch_size, (batch_id + 1) * batch_size)

                batch_id += 1
                predictions[batch_samples] = total_predictions

        self.eval()
        return predictions

    def get_autoregressive_sequence_predictions(self, test_data, data_map, encoder_states, encoder_outputs, projection_horizon):
        """Get multi-step ahead predictions autoregressively."""
        logging.info("Performing multi-step ahead prediction.")

        current_treatments = data_map['current_treatments']
        previous_treatments = data_map['previous_treatments']
        sequence_lengths = test_data['sequence_lengths'] - 1
        num_patient_points = current_treatments.shape[0]

        current_dataset = dict()
        current_dataset['current_covariates'] = np.zeros(
            shape=(num_patient_points, projection_horizon, test_data['current_covariates'].shape[-1])
        )
        current_dataset['previous_treatments'] = np.zeros(
            shape=(num_patient_points, projection_horizon, test_data['previous_treatments'].shape[-1])
        )
        current_dataset['current_treatments'] = np.zeros(
            shape=(num_patient_points, projection_horizon, test_data['current_treatments'].shape[-1])
        )
        current_dataset['init_state'] = np.zeros((num_patient_points, encoder_states.shape[-1]))

        predicted_outputs = np.zeros(shape=(num_patient_points, projection_horizon, test_data['outputs'].shape[-1]))

        for i in range(num_patient_points):
            seq_length = int(sequence_lengths[i])
            current_dataset['init_state'][i] = encoder_states[i, seq_length - 1]
            current_dataset['current_covariates'][i, 0, 0] = encoder_outputs[i, seq_length - 1]
            current_dataset['previous_treatments'][i] = previous_treatments[
                i, seq_length - 1:seq_length + projection_horizon - 1, :
            ]
            current_dataset['current_treatments'][i] = current_treatments[
                i, seq_length:seq_length + projection_horizon, :
            ]

        for t in range(projection_horizon):
            print(t)
            predictions = self.get_predictions(current_dataset)
            for i in range(num_patient_points):
                predicted_outputs[i, t] = predictions[i, t]
                if t < projection_horizon - 1:
                    current_dataset['current_covariates'][i, t + 1, 0] = predictions[i, t, 0]

        test_data['predicted_outcomes'] = predicted_outputs
        return predicted_outputs

    def evaluate_predictions(self, dataset):
        """Evaluate predictions on a dataset."""
        predictions = self.get_predictions(dataset)
        unscaled_predictions = predictions * dataset['output_stds'] + dataset['output_means']
        unscaled_predictions = unscaled_predictions.reshape(-1, self.max_sequence_length, self.num_outputs)
        unscaled_outputs = dataset['unscaled_outputs']
        active_entries = dataset['active_entries']

        mse = self.get_mse_at_follow_up_time(unscaled_predictions, unscaled_outputs, active_entries)
        mean_mse = np.mean(mse)
        return mean_mse, mse

    def get_mse_at_follow_up_time(self, prediction, output, active_entries):
        """Compute MSE at each follow-up time."""
        mses = np.sum(np.sum((prediction - output) ** 2 * active_entries, axis=-1), axis=0) / \
               active_entries.sum(axis=0).sum(axis=-1)
        return mses

    def save_network(self, model_dir, checkpoint_name):
        """Save model checkpoint."""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        save_path = os.path.join(model_dir, f"{checkpoint_name}.pt")
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_treatments': self.num_treatments,
            'num_covariates': self.num_covariates,
            'num_outputs': self.num_outputs,
            'max_sequence_length': self.max_sequence_length,
            'br_size': self.br_size,
            'rnn_hidden_units': self.rnn_hidden_units,
            'fc_hidden_units': self.fc_hidden_units,
        }, save_path)
        logging.info(f"Model saved to: {save_path}")

    def load_network(self, model_dir, checkpoint_name):
        """Load model checkpoint."""
        load_path = os.path.join(model_dir, f"{checkpoint_name}.pt")
        logging.info(f'Restoring model from {load_path}')

        checkpoint = torch.load(load_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
