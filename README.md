To accelerate training across multiple TensorFlow Sequential classification models and support faster hyperparameter tuning, threads are used to launch Docker containers on public cloud providers.

Paperspace provides a simple API for launching containers through its Gradient service.

Each API call that is made with a thread launches a containerized training job for multiple entities.

Each Docker container accesses training data that has been stored in S3 prior to training.

Metadata about the training for the entities is returned to a local client, allowing for analysis and hyperparameter tuning.

A key tradeoff involves the number of training epochs for a given entity's TensorFlow model and a loss function obtained with a validation data set.

In this case, the loss function is binary cross-entropy.
