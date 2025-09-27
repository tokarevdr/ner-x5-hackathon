import model
import os

def main():
    model.fine_tuned_model = None
    model_path = r"model"
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    parent_directory_of_script = os.path.dirname(script_directory)
    path = os.path.join(parent_directory_of_script, model_path)
    print()
    print(f"Model path {path}")
    if os.path.isdir(path):
        model.initialize(path)
        print(model.predict("Молоко"))
    else:
        raise ValueError("Directory {path} not exists")


if __name__== "__main__":
    main()