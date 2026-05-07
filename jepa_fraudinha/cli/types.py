from enum import StrEnum


class Models(StrEnum):
    LOGISTIC = "logistic"
    MLP = "mlp"
    KNN = "knn"


class AvailabeCommands(StrEnum):
    PREPARE_DATA = "prepare-data"
    TRAIN = "train"
