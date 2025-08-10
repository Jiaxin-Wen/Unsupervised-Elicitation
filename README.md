## Unsupervised Elicitation of Language Models

We introduce a new unsupervised algorithm for eliciting skills from pretrained language models. This algorithm is competitive with training on human labels on common misconceptions (TruthfulQA), math (GSM8k-verification), and helpfulness reward modeling (Alpaca). Without supervision, we train a helpful chat assistant from the Haiku 3.5 base model that outperforms a similarly trained human-supervised baseline.


<p align="center">
  <img width="100%" src="figures/llama_performance.png">
</p>

<p align="center">
  <img width="100%" src="figures/claude_performance.png">
</p>


## Setup

### Environment

1. create conda environment: `conda env create -f env.yaml`

2. install package `pip install -e .`

### API for Pretrained Base Models

You have two options for running language models:

#### Option 1: Ollama (Recommended for Local Development)

Ollama provides an easy way to run models locally:

1. Install Ollama: Visit [ollama.ai](https://ollama.ai) or run:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. Start Ollama service:
   ```bash
   ollama serve
   ```

3. Pull models:
   ```bash
   ollama pull llama3.2:latest
   ollama pull codellama:latest
   ```

4. Use the setup script for guided installation:
   ```bash
   python setup_ollama.py
   ```

See the [Ollama Integration Guide](docs/ollama_integration.md) for detailed instructions.

#### Option 2: vLLM/API Deployment

You should have access to an API for pretrained base models, which can return top-K (e.g. 20) logprobs.

Since most public api servers (e.g. openrouter) only support post-trained chat models, you probably need to deploy pretrained base models yourself. For example, we use vllm to deploy llama models in our experiments.

In particular, we highly recommend activating the `prefix caching` feature to accelerate the experiments, because our algorithm will create many API queries with similar prefixes.


### Secrets

You should create a file called SECRETS at the root of the repository with the following contents:
```
LLAMA_API_BASE=<your_api_base_url>
NYU_ORG=None
ARG_ORG=None
API_KEY=None
```

### Data Preparation

Download data from this [link](https://drive.google.com/file/d/1AJdFJO9IHfOnWHyIlGvInyndLu6EvcfV/view?usp=sharing).
Put it under the `data/` directory.

## Run

### ICM
<p align="center">
  <img width="100%" src="figures/algorithm.png">
</p>


The main script is located in `src/experiments/ICM.py`
An example command for labeling truthfulQA data:
```
cd src/experiments
python ICM.py --testbed truthfulQA --alpha 50
```

Arguments:

- `--seed`: random seed
- `--alpha`: the coefficient for mutual predictability in our scoring function
- `--testbed`: name of the testbed, e.g., alpaca, truthfulqa, gsm8k
- `--model`: name of the pretrained base model, e.g., meta-llama/Llama-3.1-70B
- `--batch_size`: size of a minibatch when running ICM on large datasets that cannot be fit in to the context all at once[^1]. 
[^1]: Since ICM relies on in-context learning, it might not be able to fix all datapoints in the context at once. In our experiments, we split the whole dataset into $N$ batches (e.g., each batch consists of 256 datapoints) based on the context limit and data length, and run ICM independently on each batch.
- `--num_seed`: number of randomly labeled datapoints in the beginning.
- `--K`: max iteration
- `--consistency_fix_K`: max iteration for consistencyfix
- `--decay`: decay rate for simulating annealing
- `--initial_T`: initial temprature for simulated annealing
- `--final_T`: final temperature for simulated annealing
- `--scheduler`: decay scheduler for simulated annealing

### Iterative Fine-tuning

Instead of using the initial pretrained model ($M_0$) to label all $N$ batches, we do iterative fine-tuning: 

- fine-tune the pretrained model on the first $j$ batches to obtain $M_j$

- use $M_j$ to label the $j+1$-th batch.

We use [axolotl](https://github.com/axolotl-ai-cloud/axolotl) for fine-tuning.

