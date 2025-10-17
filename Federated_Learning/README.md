# my-flower-app: A Flower / TensorFlow app

## Install dependencies and project

The dependencies are listed in the `pyproject.toml` and you can install them as follows:

```bash
pip install -e .
```

[// ...existing code...]

# Federated Learning Mosquito Classification - Flower Framework

## Overview

This project implements federated learning for mosquito image classification using the Flower framework, TensorFlow/Keras, and scikit-learn. The dataset is split into partitions for distributed training across clients.

## Folder Structure

- `my_flower_app/` - Main application code
  - `main_client.py` - Client entry point
  - `main_server.py` - Server entry point
  - `my_flower_app/` - Python package
    - `client_app.py` - Flower client logic
    - `server_app.py` - Flower server logic
    - `task.py` - Utilities (model, data, metrics)
    - `model/` - Pretrained and trained model files
    - `dataset/partition_{id}/` - Data partitions (train/val/test)

## Requirements

- Python 3.8+
- TensorFlow
- scikit-learn
- matplotlib
- seaborn
- Flower (`flwr`)

Install dependencies:

```powershell
pip install tensorflow scikit-learn matplotlib seaborn flwr
```

## Dataset Preparation

- Place your dataset in `my_flower_app/dataset/partition_{id}/` with subfolders for `train`, `val`, and `test` splits, each containing class folders.
- Example:
  ```
  my_flower_app/dataset/partition_0/train/Ae-aegypti/
  my_flower_app/dataset/partition_0/val/Ae-aegypti/
  my_flower_app/dataset/partition_0/test/Ae-aegypti/
  ...
  ```

## Running the Federated Learning Server

1. Open a terminal in the project root.
2. Start the server:
   ```powershell
   python my_flower_app/main_server.py
   ```
   - The server listens on `127.0.0.1:8080` by default.
   - The global model will be saved to `my_flower_app/model/mos_model_fl_trained.h5` after training.

## Running a Federated Learning Client

1. Open a new terminal for each client.
2. Start a client with its partition ID (e.g., 0 or 1):
   ```powershell
   python my_flower_app/main_client.py --partition-id 0
   ```
   - Adjust `--partition-id` for each client (e.g., 0, 1, ...).
   - Each client loads its own data partition and model weights.

## Configuration

- Training parameters (epochs, batch size, etc.) are set in `main_client.py` and `main_server.py` via the `Context` object.
- You can modify:
  - `local-epochs`: Number of epochs per client round
  - `batch-size`: Batch size for training
  - `num-server-rounds`: Number of federated rounds

## Output

- Confusion matrices and classification metrics are saved in `conf_matrix/` and `confusion_output/` after evaluation.
- Trained global model is saved in `my_flower_app/model/mos_model_fl_trained.h5`.

## Notes

- Ensure the server is running before starting clients.
- You can run multiple clients in parallel for federated training.
- For custom data partitions, update the dataset folder structure accordingly.

## References

- [Flower Documentation](https://flower.dev/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

> **Tip:** Your `pyproject.toml` file can define more than just the dependencies of your Flower app. You can also use it to specify hyperparameters for your runs and control which Flower Runtime is used. By default, it uses the Simulation Runtime, but you can switch to the Deployment Runtime when needed.
> Learn more in the [TOML configuration guide](https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html).

## Run with the Simulation Engine

In the `my-flower-app` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

## Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

You can run Flower on Docker too! Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)
