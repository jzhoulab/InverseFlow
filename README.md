# InverseFlow

This repository contains the official implementation of **Inverse Flow (IF)**.

## Overview  

Inverse generation problems, such as **denoising without ground truth observations**, pose significant challenges in scientific and real-world applications. While modern generative models (e.g., diffusion models, conditional flow matching, consistency models) have demonstrated remarkable results in conventional generation tasks, they cannot be directly applied to inverse generation problems where clean data is unavailable.  

<img width="1043" alt="image" src="https://github.com/user-attachments/assets/084ffc61-8ddf-4976-b3db-a178a1ff651f" />

**Inverse Flow (IF)** provides a framework to address this limitation by leveraging generative models for inverse generation. It supports nearly any **continuous noise distributions**, including correlated noise, and introduces two novel learning methods:  

- **Inverse Flow Matching (IFM)**
- **Inverse Consistency Model (ICM)**  

We demonstrate IF's effectiveness on both synthetic and real datasets, **outperforming prior approaches** and enabling noise models that were previously intractable. 

## Features  

- Implementation of **ICM and IFM**  
- Support for various noise types:  
  - **Gaussian noise**  
  - **Correlated Gaussian noise**
  - **Poisson noise**  
  - **Reversing the Jacobi process** 
  - ... 
- Jupyter Notebook for running the **toy model**  

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

## Usage
To run the main script, use:

```sh
python run.py
```

You can also explore the toy model in the Jupyter notebook:
```sh
jupyter notebook toy_model.ipynb
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.
