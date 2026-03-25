# Demo Specification

## Live Demo Flow

1. Select a machine type.
2. Upload a `.wav` file.
3. Send the audio to the backend for inference.
4. Return:
   - anomaly score
   - normal or anomalous verdict
   - spectrogram image
   - highlighted region or explanation overlay
5. Optionally compare with a known normal reference sample.

## What Should Impress In Person

- quick response time
- clean upload flow
- obvious score difference between normal and anomalous clips
- side-by-side sample comparisons
- visual explanation rather than just a number

## Stretch Goals

- sample gallery
- machine-type metrics page
- saved inference history for the session
