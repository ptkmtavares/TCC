import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from dataExtractor import getTrainingTestSet
from mlp import MLP, train_mlp

def train_mlp_tune(config, data, labels, test_data, test_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalizar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_set_normalized = scaler.fit_transform(data)
    test_set_normalized = scaler.transform(test_data)

    X_train = torch.tensor(train_set_normalized, dtype=torch.float32).to(device)
    y_train = torch.tensor(labels, dtype=torch.long).to(device)
    X_test = torch.tensor(test_set_normalized, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_labels, dtype=torch.long).to(device)

    input_dim = data.shape[1]
    hidden_dim = config["hidden_dim"]
    output_dim = 3

    model = MLP(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    val_loss = train_mlp(model, criterion, optimizer, X_train, y_train, X_test, y_test, num_epochs=config["num_epochs"], patience=config["patience"], printInfo=False)
    
    session.report({"val_loss": val_loss})

def trial_dirname_creator(trial):
    return f"trial_{trial.trial_id}"

def getHyperparameters(train_set, test_set, train_labels, test_labels):
    config = {
        "hidden_dim": tune.choice([32, 64, 128]),
        "lr": tune.loguniform(1e-4, 1e-3),
        "weight_decay": tune.loguniform(1e-2, 1e-1),
        "num_epochs": tune.lograndint(1000, 5000),
        "patience": tune.randint(150, 350)
    }

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=100,
        grace_period=10,
        reduction_factor=2
    )

    analysis = tune.run(
        tune.with_parameters(train_mlp_tune, data=train_set, labels=train_labels, test_data=test_set, test_labels=test_labels),
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        trial_dirname_creator=trial_dirname_creator,
        verbose=1
    )

    return analysis.get_best_config(metric="val_loss", mode="min")

def main():
    # Carregar e pré-processar os dados
    selected_data = ['ham', 'spam', 'phishing']
    data, index = getTrainingTestSet('Dataset/index', selected_data, 1.0)
    train_set, test_set, train_labels, test_labels = train_test_split(data, index, train_size=0.75, random_state=9, shuffle=True)

    config = {
        "hidden_dim": tune.choice([128, 256, 512, 1024]),
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "num_epochs": tune.lograndint(1000, 10000),
        "patience": tune.randint(150, 300)
    }

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=100,
        grace_period=10,
        reduction_factor=2
    )

    analysis = tune.run(
        tune.with_parameters(train_mlp_tune, data=train_set, labels=train_labels, test_data=test_set, test_labels=test_labels),
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        trial_dirname_creator=trial_dirname_creator
    )

    #print("Best config: ", analysis.get_best_config(metric="val_loss", mode="min"))
    return analysis.get_best_config(metric="val_loss", mode="min")