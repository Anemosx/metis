# Metis - Machine Learning Experimentation and Training Interactive Suite

![MIT license](https://img.shields.io/badge/license-MIT-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Black Linting](https://github.com/Anemosx/metis/actions/workflows/lint.yml/badge.svg)](https://github.com/Anemosx/metis/actions/workflows/lint.yml)
[![PyTest](https://github.com/Anemosx/metis/actions/workflows/test.yml/badge.svg)](https://github.com/Anemosx/metis/actions/workflows/test.yml)

![metis_banner](docs/metis.png)

Metis is an interactive project designed to provide a hands-on learning experience in machine learning. 
From foundational code to user-friendly interfaces, it allows you to explore machine learning concepts in-depth. 
Whether you're learning about reinforcement learning or advanced techniques like diffusion models, 
Metis offers a comprehensive journey into the nuances and intricacies of various machine learning techniques.

## Installation

**Install Project Dependencies**
First ensure you have Poetry installed for managing project dependencies. Then run the following commands:
```
pip install poetry
poetry install
```

## Running the Project

1. **Running Machine Learning Models**

    To experiment with individual machine learning models, navigate to the appropriate folder under `src/` and execute:
    ```
    python src/.../train.py
    ```

2. **Running the Pretrained Model UI**

    For pretrained models with a user interface, run the following command:
    ```
    python app/main.py
    ```

    Once the server is running, open your web browser and go to:
    [http://localhost:8000](http://localhost:8000)

    To view specific examples, append the relevant topic to the URL. For example, to explore the MNIST example, navigate to:
    [http://localhost:8000/mnist](http://localhost:8000/mnist)
