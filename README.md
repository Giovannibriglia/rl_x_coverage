
## About the Project
This project implements MARL algorithms trained based on Voronoi-based Tesselation.

## 1. Installation
1. Create a new python virtual environment (with 'python > 3.10'):
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   > The use of a virtual environment to keep dependencies isolated, is not mandatory, but helps avoid contaminating the global Python environment.
   > If you create one, be sure to activate it before using any of this repository’s functionality.
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Install setup:
   ```
   python setup.py install
   ```

## 2. Launch

Launch `__main__.py


## 3. Set of Experiments
| Experimental Batch         | Train scenario                            | Test Scenario                                           | Done   |
|----------------------------|-------------------------------------------|---------------------------------------------------------|--------|
| #1: basic1                 | Basic Env  <br/> 3 agents <br/> 1 Gauss   | Basic Env  <br/> 3 agents <br/> 1 Gauss                 |        |
|                            |                                           | Basic Env  <br/> 5 agents <br/> 1 Gauss                 |        |
|                            |                                           | Basic Env  <br/> 7 agents <br/> 1 Gauss                 |        |
|                            |                                           | Basic Env  <br/> 3 agents <br/> 3 Gauss                 |        |
|                            |                                           | Basic Env  <br/> 5 agents <br/> 5 Gauss                 |        |
|                            |                                           | Basic Env  <br/> 7 agents <br/> 7 Gauss                 |        |
| #2: basic2                 | Basic Env  <br/> 3 agents <br/> 3 Gauss   | Basic Env  <br/> 3 agents <br/> 3 Gauss                 |        |
|                            |                                           | Basic Env  <br/> 5 agents <br/> 5 Gauss                 |        |
|                            |                                           | Basic Env  <br/> 7 agents <br/> 7 Gauss                 |        |
| #3: dynamic env            | Basic Env  <br/> 3 agents <br/> 3 Gauss   | Dynamic Env  <br/> 3 agents <br/> 1 Gauss               |        |
|                            |                                           | Dynamic Env  <br/> 5 agents <br/> 1 Gauss               |        |
|                            |                                           | Dynamic Env  <br/> 7 agents <br/> 1 Gauss               |        |
|                            |                                           | Dynamic Env  <br/> 3 agents <br/> 3 Gauss               |        |
|                            |                                           | Dynamic Env  <br/> 5 agents <br/> 5 Gauss               |        |
|                            |                                           | Dynamic Env  <br/> 7 agents <br/> 7 Gauss               |        |
| #4: partial obs            | Basic Env  <br/> 3 agents <br/> 3 Gauss   | Partial f.o.v.  <br/> 3 agents <br/> 1 Gauss            |        |
|                            |                                           | Partial f.o.v.  <br/> 5 agents <br/> 1 Gauss            |        |
|                            |                                           | Partial f.o.v.  <br/> 7 agents <br/> 1 Gauss            |        |
|                            |                                           | Partial f.o.v.  <br/> 3 agents <br/> 3 Gauss            |        |
|                            |                                           | Partial f.o.v.  <br/> 5 agents <br/> 5 Gauss            |        |
|                            |                                           | Partial f.o.v.  <br/> 7 agents <br/> 7 Gauss            |        |
| #5: easy non convex envs   | Basic Env  <br/> 3 agents <br/> 3 Gauss   | Env with some obstacles  <br/> 3 agents <br/> 1 Gauss   |        |
|                            |                                           | Env with some obstacles  <br/> 5 agents <br/> 1 Gauss   |        |
|                            |                                           | Env with some obstacles  <br/> 7 agents <br/> 1 Gauss   |        |
|                            |                                           | Env with some obstacles  <br/> 3 agents <br/> 3 Gauss   |        |
|                            |                                           | Env with some obstacles  <br/> 5 agents <br/> 5 Gauss   |        |
|                            |                                           | Env with some obstacles  <br/> 7 agents <br/> 7 Gauss   |        |
| #6: hard non convex envs   | Basic Env  <br/> 3 agents <br/> 3 Gauss   | Env L-like  <br/> 3 agents <br/> 1 Gauss                |        |
|                            |                                           | Env L-like   <br/> 5 agents <br/> 1 Gauss               |        |
|                            |                                           | Env L-like   <br/> 7 agents <br/> 1 Gauss               |        |
|                            |                                           | Env L-like   <br/> 3 agents <br/> 3 Gauss               |        |
|                            |                                           | Env L-like   <br/> 5 agents <br/> 5 Gauss               |        |
|                            |                                           | Env L-like  <br/> 7 agents <br/> 7 Gauss                |        |
