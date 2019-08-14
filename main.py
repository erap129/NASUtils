from config import init_configurations
from abstract_models_generation import *
from pytorch_model_generation import *


# ======================================================================================================================

def main():
    init_configurations(False, 10, 10, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0.5, 10, 300, 300, 3, 2)
    pop = initialize_population()
    model = pop[0]
    model = finalize_model(model)
    pytorch_model = create_pytorch_model(model, apply_fix=True)
    print(pytorch_model)


if __name__ == "__main__":
    main()
