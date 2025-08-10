# Running Experiments with Ollama Models

This guide explains how to run the ICM (Iterative Consistency Maximization) experiments using local Ollama models instead of API-based models.

## Prerequisites

### 1. Install Ollama

Visit [https://ollama.ai](https://ollama.ai) and follow the installation instructions for your platform.

**For Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**For Windows:**
Download and run the installer from the Ollama website.

### 2. Start Ollama Service

```bash
ollama serve
```

Keep this running in a separate terminal.

### 3. Pull Required Models

Pull some models to use for experiments:

```bash
# Small models (good for testing)
ollama pull llama3.2:1b
ollama pull llama3.2:3b

# Medium models (better performance)
ollama pull llama3.2:latest      # 8B parameters
ollama pull mistral:7b

# Large models (best performance, requires more RAM)
ollama pull llama3.1:70b         # Requires ~40GB RAM
ollama pull mixtral:8x7b         # Requires ~26GB RAM
```

### 4. Verify Installation

Run the verification script:

```bash
python verify_ollama_integration.py
```

## Running Experiments

### Basic Usage

Run the ICM experiment with a local Ollama model:

```bash
python src/experiments/ICM.py \
    --model llama3.2:latest \
    --testbed gsm8k \
    --batch_size 64 \
    --num_seed 8 \
    --K 1000 \
    --use_ollama
```

### Model Selection

Choose models based on your hardware:

**For systems with 8-16GB RAM:**
```bash
--model llama3.2:1b    # Fastest, lowest quality
--model llama3.2:3b    # Good balance
```

**For systems with 16-32GB RAM:**
```bash
--model llama3.2:latest  # 8B model, good quality
--model mistral:7b       # Alternative 7B model
```

**For systems with 32GB+ RAM:**
```bash
--model llama3.1:70b     # High quality, slow
--model mixtral:8x7b     # Good alternative
```

### Testbed Options

Available testbeds with example commands:

**GSM8K (Math Problems):**
```bash
python src/experiments/ICM.py \
    --model llama3.2:latest \
    --testbed gsm8k \
    --batch_size 64 \
    --num_seed 8
```

**TruthfulQA:**
```bash
python src/experiments/ICM.py \
    --model llama3.2:latest \
    --testbed truthfulQA \
    --batch_size 64 \
    --num_seed 8
```

**Alpaca (Instruction Following):**
```bash
python src/experiments/ICM.py \
    --model llama3.2:latest \
    --testbed alpaca \
    --batch_size 64 \
    --num_seed 8
```

### Configuration Options

**Ollama-specific options:**
- `--ollama_host`: Ollama server URL (default: http://localhost:11434)
- `--ollama_temperature`: Temperature for generation (default: 0.0)
- `--use_ollama`: Force Ollama usage (auto-detected for most local models)

**Experiment parameters:**
- `--alpha`: Weight for training probability vs consistency (default: 30)
- `--batch_size`: Number of examples to process (default: 256)
- `--num_seed`: Number of initial labeled examples (default: 8)
- `--K`: Maximum number of iterations (default: 3000)
- `--consistency_fix_K`: Iterations for consistency fixing (default: 10)

## Performance Tips

### 1. Reduce Batch Size
For local models, use smaller batch sizes to reduce memory usage:

```bash
--batch_size 32    # Instead of 256
--batch_size 16    # For very constrained systems
```

### 2. Reduce Iterations
Use fewer iterations for faster testing:

```bash
--K 500           # Instead of 3000
--consistency_fix_K 5  # Instead of 10
```

### 3. Monitor Resource Usage

```bash
# Monitor GPU usage (if using GPU)
nvidia-smi -l 1

# Monitor RAM usage
htop
```

## Example Full Commands

**Quick test (small model, few iterations):**
```bash
python src/experiments/ICM.py \
    --model llama3.2:1b \
    --testbed gsm8k \
    --batch_size 16 \
    --num_seed 4 \
    --K 100 \
    --use_ollama
```

**Production run (good balance):**
```bash
python src/experiments/ICM.py \
    --model llama3.2:latest \
    --testbed gsm8k \
    --batch_size 64 \
    --num_seed 8 \
    --K 1000 \
    --alpha 30 \
    --use_ollama
```

**High-quality run (requires powerful hardware):**
```bash
python src/experiments/ICM.py \
    --model llama3.1:70b \
    --testbed truthfulQA \
    --batch_size 32 \
    --num_seed 8 \
    --K 2000 \
    --alpha 50 \
    --use_ollama
```

## Troubleshooting

### Common Issues

**1. "Ollama service may not be available"**
- Ensure `ollama serve` is running
- Check that port 11434 is not blocked

**2. "Model not found"**
- Pull the model: `ollama pull model_name`
- List available models: `ollama list`

**3. Out of memory errors**
- Use smaller models (llama3.2:1b instead of llama3.1:70b)
- Reduce batch_size
- Close other applications

**4. Slow performance**
- Ensure you have sufficient RAM for the model
- Consider using GPU if available
- Use smaller models for testing

### Debugging

Enable debug logging:

```bash
export PYTHONPATH=/home/dariast/Unsupervised-Elicitation:$PYTHONPATH
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
" src/experiments/ICM.py --model llama3.2:latest --testbed gsm8k
```

## Output

Results are saved to:
- `log_{experiment_name}.jsonl`: Progress log with metrics per iteration
- Console output: Real-time progress and metrics

Example log entry:
```json
{"iter": 0, "flip_cnt": 1, "acc": 0.75, "score": 22.5}
```

Where:
- `iter`: Current iteration
- `flip_cnt`: Number of label flips
- `acc`: Accuracy on training data
- `score`: Energy function value (alpha * prob - inconsistencies)
