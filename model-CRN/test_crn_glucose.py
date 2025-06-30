# Copyright (c) 2020, Ioana Bica
# Modified for glucose data

import os
import argparse
import logging

from CRN_glucose_evaluate import test_CRN_encoder_glucose, test_CRN_decoder_glucose
from utils.glucose_simulation import get_glucose_sim_data


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='../Data/ml_dataset.csv', type=str,
                       help="Path to the glucose dataset CSV file")
    parser.add_argument("--sequence_length", default=20, type=int,
                       help="Length of input sequences")
    parser.add_argument("--prediction_horizon", default=5, type=int,
                       help="Number of steps ahead to predict")
    parser.add_argument("--results_dir", default='results')
    parser.add_argument("--model_name", default="crn_glucose_test")
    parser.add_argument("--b_encoder_hyperparm_tuning", default=False, type=bool)
    parser.add_argument("--b_decoder_hyperparm_tuning", default=False, type=bool)
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    
    # Load glucose data instead of cancer simulation
    pickle_map = get_glucose_sim_data(
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon
    )

    encoder_model_name = 'encoder_' + args.model_name
    encoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, encoder_model_name)

    models_dir = '{}/crn_models'.format(args.results_dir)
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    # Train and evaluate encoder
    rmse_encoder = test_CRN_encoder_glucose(pickle_map=pickle_map, models_dir=models_dir,
                                           encoder_model_name=encoder_model_name,
                                           encoder_hyperparams_file=encoder_hyperparams_file,
                                           b_encoder_hyperparm_tuning=args.b_encoder_hyperparm_tuning)

    # Train and evaluate decoder
    decoder_model_name = 'decoder_' + args.model_name
    decoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, decoder_model_name)

    max_projection_horizon = args.prediction_horizon
    projection_horizon = args.prediction_horizon
    
    rmse_decoder = test_CRN_decoder_glucose(pickle_map=pickle_map, max_projection_horizon=max_projection_horizon,
                                           projection_horizon=projection_horizon,
                                           models_dir=models_dir,
                                           encoder_model_name=encoder_model_name,
                                           encoder_hyperparams_file=encoder_hyperparams_file,
                                           decoder_model_name=decoder_model_name,
                                           decoder_hyperparams_file=decoder_hyperparams_file,
                                           b_decoder_hyperparm_tuning=args.b_decoder_hyperparm_tuning)

    logging.info("Glucose CRN Model Results")
    logging.info(f"Data path: {args.data_path}")
    logging.info(f"Sequence length: {args.sequence_length}")
    logging.info(f"Prediction horizon: {args.prediction_horizon}")
    
    print("RMSE for one-step-ahead glucose prediction:")
    print(rmse_encoder)

    print(f"RMSE for {args.prediction_horizon}-step-ahead glucose prediction:")
    print(rmse_decoder)