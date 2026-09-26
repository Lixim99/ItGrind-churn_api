class DataPreparationError(Exception):
    def __init__(
        self,
        message: str,
        details=None,
    ):
        self.message = message
        self.details = details

        super().__init__(message)


class ModelPredictionError(Exception):
    def __init__(
        self,
        message: str,
        details=None,
    ):
        self.message = message
        self.details = details

        super().__init__(message)


class ModelNotFoundError(Exception):
    ...


class EmptyDatasetError(Exception):
    ...
