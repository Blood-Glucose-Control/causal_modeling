# Copyright (c) 2020, Ioana Bica

import os
import argparse
import logging

from CRN_encoder_evaluate import test_CRN_encoder
from CRN_decoder_evaluate import test_CRN_decoder
from utils.cancer_simulation import get_cancer_sim_data
from utils.diabetes_data_generator import get_diabetes_sim_data


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemo_coeff", default=2, type=int)
    parser.add_argument("--radio_coeff", default=2, type=int)
    parser.add_argument("--results_dir", default='results')
    parser.add_argument("--model_name", default="crn_test_2")
    parser.add_argument("--b_encoder_hyperparm_tuning", default=False)
    parser.add_argument("--b_decoder_hyperparm_tuning", default=False)
    
    # New parameters for insulin encoding
    parser.add_argument("--treatment_encoding", default="onehot", choices=["onehot", "integer"],
                       help="Treatment encoding strategy: 'onehot' for cancer, 'integer' for diabetes")
    parser.add_argument("--num_dose_levels", default=5, type=int,
                       help="Number of dose levels for integer encoding (diabetes)")
    parser.add_argument("--data_type", default="cancer", choices=["cancer", "diabetes"],
                       help="Type of data to use: 'cancer' or 'diabetes'")
    
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    
    # Load data based on type
    if args.data_type == "cancer":
        pickle_map = get_cancer_sim_data(chemo_coeff=args.chemo_coeff, radio_coeff=args.radio_coeff, 
                                        b_load=False, b_save=False, model_root=args.results_dir)
        # Ensure backward compatibility for cancer models
        if args.treatment_encoding == "integer":
            logging.warning("Integer encoding requested for cancer data. Switching to one-hot for compatibility.")
            args.treatment_encoding = "onehot"
    else:  # diabetes
        logging.info("Diabetes data type selected. Generating real diabetes data using data-api...")
        pickle_map = get_diabetes_sim_data(
            total_days=21,  # 3 weeks of patient data
            window_days=7,  # 1 week training windows
            seed=42,
            b_load=False, 
            b_save=False, 
            model_root=args.results_dir
        )
        # Force integer encoding for diabetes
        args.treatment_encoding = "integer"
        logging.info(f"✓ Generated diabetes dataset with {pickle_map['training_data']['glucose'].shape[0]} training windows")

    encoder_model_name = 'encoder_' + args.model_name
    encoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, encoder_model_name)

    models_dir = '{}/crn_models'.format(args.results_dir)
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    rmse_encoder = test_CRN_encoder(pickle_map=pickle_map, models_dir=models_dir,
                                    encoder_model_name=encoder_model_name,
                                    encoder_hyperparams_file=encoder_hyperparams_file,
                                    b_encoder_hyperparm_tuning=args.b_encoder_hyperparm_tuning,
                                    treatment_encoding=args.treatment_encoding,
                                    num_dose_levels=args.num_dose_levels)


    decoder_model_name = 'decoder_' + args.model_name
    decoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, decoder_model_name)

    """
    The counterfactual test data for a sequence of treatments in the future was simulated for a 
    projection horizon of 5 timesteps. 
   
    """

    max_projection_horizon = 5
    projection_horizon = 5
    
    rmse_decoder = test_CRN_decoder(pickle_map=pickle_map, max_projection_horizon=max_projection_horizon,
                                    projection_horizon=projection_horizon,
                                    models_dir=models_dir,
                                    encoder_model_name=encoder_model_name,
                                    encoder_hyperparams_file=encoder_hyperparams_file,
                                    decoder_model_name=decoder_model_name,
                                    decoder_hyperparams_file=decoder_hyperparams_file,
                                    b_decoder_hyperparm_tuning=args.b_decoder_hyperparm_tuning,
                                    treatment_encoding=args.treatment_encoding,
                                    num_dose_levels=args.num_dose_levels)

    logging.info("Chemo coeff {} | Radio coeff {}".format(args.chemo_coeff, args.radio_coeff))
    print("RMSE for one-step-ahead prediction.")
    print(rmse_encoder)

    print("Results for 5-step-ahead prediction.")
    print(rmse_decoder)
