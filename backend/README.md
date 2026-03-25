# Backend Plan

The backend will be a small FastAPI service that:

- accepts audio uploads
- preprocesses them into spectrograms
- runs model inference
- returns anomaly score, label, and visualization payloads
