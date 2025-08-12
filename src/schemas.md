### MongoDB Schemas

- `contests`
```json
{
  "contest": 2805,
  "date": "2025-08-10",
  "numbers": [x1,x2,...,x20],
  "raw": {...},
  "created_at": "ISODate(...)"
}
```

- `predictions`
```json
{
  "generated_at": "ISODate(...)",
  "model_version": "v2025-08-11-1",
  "p_i": { "1":0.02,"2":0.015,"...":"..."},
  "candidate_pool": [..],
  "tickets": [[50 nums], [50 nums], [50 nums]],
  "simulations": { "n":100000, "p_at_least_18":0.003, "p_at_least_17":0.02, "p_at_least_16":0.12 },
  "confidence_passed": false,
  "criteria": {"thresholds": {"18":0.02,"17":0.10,"16":0.50}}
}
```