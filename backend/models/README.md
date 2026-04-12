# Backend Models

The web demo loads these six artifacts at startup:

| Machine | Required files |
|---|---|
| Fan id_06 | `fan_id_06_mlp_best.pth`, `fan_id_06_normalizer.npz` |
| Pump id_04 | `pump_id_04_mlp_best.pth`, `pump_id_04_normalizer.npz` |
| Valve id_04 | `valve_id_04_mlp_best.pth`, `valve_id_04_normalizer.npz` |

These are the selected best demo models from the final per-ID v2 training run. They are small enough for the class demo repository, but they should be treated as generated artifacts rather than source code.
