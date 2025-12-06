<div style="text-align: center; display: flex; align-items: center; justify-content: center; background-color: white; padding: 20px; border-radius: 30px;">
  <img src="./static/ASC.jpg" alt="AgentSociety Challenge Logo" width="100" style="margin-right: 20px; border-radius: 10%;">
  <h1 style="color: black; margin: 0; font-size: 2em;">WWW'25 AgentSociety Challenge: WebSocietySimulator</h1>
</div>

## Presentation Slides and Experiment Logs

Our presentation slides and experiment logs are available in the additional_files folder.

## Quick Start (Instructions to Run Our Model)

### 1. Install the Library

The repository is organized using [Python Poetry](https://python-poetry.org/). Follow these steps to install the library:

1. Clone the repository:
   ```bash
   git clone <this_repo>
   cd websocietysimulator
   ```

2. Install dependencies:
  - Option 1: Install dependencies using Poetry: (Recommended)
    ```bash
    poetry install  && \
    poetry shell
    ```
  - Option 2: Install dependencies using pip(COMING SOON):
    ```bash
    pip install websocietysimulator
    ```
  - Option 3: Install dependencies using conda:
    ```bash
    conda create -n websocietysimulator python=3.11 && \
    conda activate websocietysimulator && \
    pip install -r requirements.txt && \
    pip install .
    ```

3. Verify the installation:
   ```python
   import websocietysimulator
   ```

---

### 2. Data Preparation

1. Download the raw dataset from Yelp[1].
2. Run the `data_process.py` script to process the dataset:
   ```bash
   mkdir -p data/yelp_processed
   python data_process.py --input <path_to_raw_dataset> --output <path_to_repo>/data/yelp_processed
   ```
- Check out the [Data Preparation Guide](./tutorials/data_preparation.md) for more information.
- **NOTICE: You Need at least 16GB RAM to process the dataset.**

### 3. Running Models

Prior to running the program, you must modify the MyRecommendationAgent class 
```python
class MyRecommendationAgent(RecommendationAgent):

  llm = GeminiLLM(api_key="YOUR_API_KEY")
```
substituting the api_key field with a valid Gemini API key.

You must also ensure that the google-generativeai package is installed, as the installation script does not install this by default.
```bash
pip install google-generativeai
```

To run the baseline model (assuming you are in the agentsocietychallenge directory):
```bash
cd example
python RecAgent_baseline.py
```
To run our model (assuming you are in the agentsocietychallenge directory):
```bash
cd example
python RecAgent.py
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## References

[1] Yelp Dataset: https://www.yelp.com/dataset
