# Roadmap

## Phase 1: Data and Preprocessing

- validate dataset layout
- build metadata manifest
- load audio safely and consistently
- generate log-mel spectrograms

## Phase 2: Baseline Model

- train a convolutional autoencoder on normal clips only
- compute anomaly scores from reconstruction error
- establish machine-type-specific thresholds

## Phase 3: Improved Model

- compare a stronger CNN-based anomaly model
- evaluate per machine type
- analyze false positives and false negatives

## Phase 4: Demo and Presentation

- expose inference through FastAPI
- connect the frontend upload flow
- visualize spectrograms and anomaly explanations
- prepare benchmark clips for the live demo
