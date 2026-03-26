from predictor import NBAPredictor


if __name__ == "__main__":
    predictor = NBAPredictor()

    if predictor.needs_retraining():
        predictor.train_model()
        predictor.save_model()
    else:
        predictor.logger.info("Model is up-to-date. Skipping training.")