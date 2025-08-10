# Running ICM Experiments with Ollama Models

This guide provides comprehensive instructions for running Iterative Consistency Maximization (ICM) experiments using local Ollama models instead of API-based models like OpenAI or Anthropic.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Model Selection Guide](#model-selection-guide)
- [Running Experiments](#running-experiments)
- [Configuration Options](#configuration-options)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## Prerequisites

### 1. Install Ollama

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download and install from [https://ollama.ai](https://ollama.ai)

**Docker (Alternative):**
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### 2. Start Ollama Service

```bash
ollama serve
```

Keep this running in a separate terminal throughout your experiments.

### 3. Install Python Dependencies

Make sure you have the required packages:
```bash
pip install ollama langchain-ollama langchain-core
```

### 4. Verify Installation

Run the verification script:
```bash
python verify_ollama_integration.py
```

If successful, test the example:
```bash
python examples/ollama_example.py
```

## Quick Start

### 1. Pull a Model
```bash
# For testing (small, fast)
ollama pull llama3.2:1b

# For production (good balance)
ollama pull llama3.2:latest

# For best quality (requires more resources)
ollama pull llama3.1:70b
```

### 2. Run a Quick Test
```bash
python src/experiments/ICM.py \
    --model llama3.2:latest \
    --testbed gsm8k \
    --batch_size 16 \
    --num_seed 4 \
    --K 100
```

### 3. Check Results
Results are saved to `log_{experiment_name}.jsonl` and displayed in real-time.

## Model Selection Guide

Choose models based on your hardware capabilities:

### System Requirements by Model Size

| Model | Parameters | RAM Required | Speed | Quality |
|-------|------------|--------------|-------|---------|
| `llama3.2:1b` | 1B | 2GB | Very Fast | Basic |
| `llama3.2:3b` | 3B | 4GB | Fast | Good |
| `llama3.2:latest` | 8B | 8GB | Medium | Very Good |
| `mistral:7b` | 7B | 8GB | Medium | Very Good |
| `llama3.1:70b` | 70B | 40GB | Slow | Excellent |
| `mixtral:8x7b` | 47B | 26GB | Slow | Excellent |

### Recommended Models by Use Case

**Development/Testing:**
```bash
ollama pull llama3.2:1b
ollama pull llama3.2:3b
```

**Production Experiments:**
```bash
ollama pull llama3.2:latest
ollama pull mistral:7b
```

**Research/Best Quality:**
```bash
ollama pull llama3.1:70b
ollama pull mixtral:8x7b
```

## Running Experiments

### Basic Experiment Commands

**GSM8K (Math Reasoning):**
```bash
python src/experiments/ICM.py \
    --model llama3.2:latest \
    --testbed gsm8k \
    --batch_size 64 \
    --num_seed 8 \
    --K 1000 \
    --alpha 30
```

**TruthfulQA (Truthfulness):**
```bash
python src/experiments/ICM.py \
    --model llama3.2:latest \
    --testbed truthfulQA \
    --batch_size 64 \
    --num_seed 8 \
    --K 1000 \
    --alpha 30
```

**TruthfulQA Preference (Comparative):**
```bash
python src/experiments/ICM.py \
    --model llama3.2:latest \
    --testbed truthfulQA-preference \
    --batch_size 32 \
    --num_seed 8 \
    --K 1000 \
    --alpha 30
```

**Alpaca (Instruction Following):**
```bash
python src/experiments/ICM.py \
    --model llama3.2:latest \
    --testbed alpaca \
    --batch_size 32 \
    --num_seed 8 \
    --K 1000 \
    --alpha 30
```

### Experiment Scaling by Hardware

**Low-Resource Systems (8-16GB RAM):**
```bash
python src/experiments/ICM.py \
    --model llama3.2:1b \
    --testbed gsm8k \
    --batch_size 16 \
    --num_seed 4 \
    --K 500 \
    --consistency_fix_K 5 \
    --alpha 20
```

**Medium Systems (16-32GB RAM):**
```bash
python src/experiments/ICM.py \
    --model llama3.2:latest \
    --testbed gsm8k \
    --batch_size 64 \
    --num_seed 8 \
    --K 1000 \
    --consistency_fix_K 10 \
    --alpha 30
```

**High-Resource Systems (32GB+ RAM):**
```bash
python src/experiments/ICM.py \
    --model llama3.1:70b \
    --testbed truthfulQA \
    --batch_size 32 \
    --num_seed 8 \
    --K 2000 \
    --consistency_fix_K 10 \
    --alpha 50
```

## Configuration Options

### Core Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `--model` | Ollama model name | `meta-llama/Llama-3.1-70B` | See model guide |
| `--testbed` | Evaluation dataset | `gsm8k` | `gsm8k`, `truthfulQA`, `alpaca` |
| `--batch_size` | Number of examples | 256 | 16-256 |
| `--num_seed` | Initial labeled examples | 8 | 4-16 |
| `--K` | Maximum iterations | 3000 | 100-5000 |
| `--alpha` | Consistency weight | 30 | 10-100 |

### Ollama-Specific Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--ollama_host` | Ollama server URL | `http://localhost:11434` |
| `--ollama_temperature` | Generation temperature | 0.0 |
| `--use_ollama` | Force Ollama usage | Auto-detected |

### Advanced Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--consistency_fix_K` | Consistency iterations | 10 | Reduce for speed |
| `--decay` | Temperature decay | 0.99 | For simulated annealing |
| `--initial_T` | Initial temperature | 10 | Annealing start |
| `--final_T` | Final temperature | 0.01 | Annealing end |
| `--scheduler` | Annealing schedule | `log` | `log` or `exp` |

## Performance Optimization

### Speed Optimization

1. **Use Smaller Models:**
   ```bash
   --model llama3.2:1b  # Instead of llama3.1:70b
   ```

2. **Reduce Batch Size:**
   ```bash
   --batch_size 16      # Instead of 256
   ```

3. **Fewer Iterations:**
   ```bash
   --K 500              # Instead of 3000
   --consistency_fix_K 5 # Instead of 10
   ```

4. **GPU Acceleration (if available):**
   ```bash
   # Ollama automatically uses GPU if available
   nvidia-smi  # Check GPU usage
   ```

### Memory Optimization

1. **Monitor Usage:**
   ```bash
   htop                 # Monitor RAM
   nvidia-smi           # Monitor GPU memory
   ```

2. **Close Other Applications:**
   ```bash
   # Close browsers, IDEs, etc. before running large models
   ```

3. **Use Swap (if needed):**
   ```bash
   # Ensure adequate swap space for large models
   free -h
   ```

## Troubleshooting

### Common Issues and Solutions

**1. "Ollama service may not be available"**
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama
ollama serve

# Check port availability
netstat -tuln | grep 11434
```

**2. "Model not found"**
```bash
# List available models
ollama list

# Pull the model
ollama pull llama3.2:latest

# Verify model availability
ollama list | grep llama3.2
```

**3. "Connection refused"**
```bash
# Check Ollama status
curl http://localhost:11434/api/version

# Restart Ollama
pkill ollama
ollama serve
```

**4. "Error in predict_assignment: 'metadata'"**
```bash
# This indicates a response parsing issue - usually resolved by:
# 1. Ensuring models are properly loaded
# 2. Restarting the experiment
# 3. Using a different model if the issue persists

# Try with a different model:
python src/experiments/ICM.py \
    --model llama3.2:1b \
    --testbed gsm8k \
    --batch_size 8 \
    --K 50
```

**5. Out of Memory Errors**
```bash
# Use smaller model
--model llama3.2:1b

# Reduce batch size
--batch_size 8

# Check available memory
free -h
```

**6. Slow Performance**
```bash
# Check CPU/GPU usage
htop
nvidia-smi

# Use faster model
--model llama3.2:3b  # Instead of llama3.1:70b

# Reduce problem size
--K 100 --batch_size 16
```

### Debug Mode

Enable detailed logging:
```bash
export PYTHONPATH=/home/dariast/Unsupervised-Elicitation:$PYTHONPATH
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
exec(open('src/experiments/ICM.py').read())
" --model llama3.2:latest --testbed gsm8k --K 10
```

### Health Checks

**Check Ollama Service:**
```bash
curl http://localhost:11434/api/tags
```

**Test Model:**
```bash
ollama run llama3.2:latest "What is 2+2?"
```

**Verify Integration:**
```bash
python verify_ollama_integration.py
```

## Examples

### Example 1: Quick Development Test
```bash
# Fast test with small model
python src/experiments/ICM.py \
    --model llama3.2:1b \
    --testbed gsm8k \
    --batch_size 8 \
    --num_seed 2 \
    --K 50 \
    --alpha 10
```

### Example 2: Standard Research Experiment
```bash
# Balanced experiment
python src/experiments/ICM.py \
    --model llama3.2:latest \
    --testbed truthfulQA \
    --batch_size 64 \
    --num_seed 8 \
    --K 1000 \
    --alpha 30 \
    --consistency_fix_K 10
```

### Example 3: High-Quality Experiment
```bash
# Best quality (requires powerful hardware)
python src/experiments/ICM.py \
    --model llama3.1:70b \
    --testbed gsm8k \
    --batch_size 32 \
    --num_seed 8 \
    --K 2000 \
    --alpha 50 \
    --consistency_fix_K 15
```

### Example 4: Comparative Study
```bash
# Run multiple models for comparison
for model in llama3.2:1b llama3.2:latest mistral:7b; do
    python src/experiments/ICM.py \
        --model $model \
        --testbed gsm8k \
        --batch_size 32 \
        --num_seed 8 \
        --K 500 \
        --seed 42  # Same seed for fair comparison
done
```

## Output and Results

### Log Files
- **Real-time progress:** Console output with metrics
- **Detailed logs:** `log_{experiment_name}.jsonl`
- **Final results:** Summary statistics at completion

### Log Format
```json
{
  "iter": 0,
  "flip_cnt": 1,
  "acc": 0.75,
  "score": 22.5
}
```

Where:
- `iter`: Current iteration number
- `flip_cnt`: Number of label flips performed
- `acc`: Accuracy on training examples
- `score`: Energy function value (alpha * prob - inconsistencies)

### Monitoring Progress
```bash
# Watch log file in real-time
tail -f log_gsm8k-llama3_2_latest-*.jsonl

# Plot progress (if you have plotting tools)
python plot_progress.py log_gsm8k-llama3_2_latest-*.jsonl
```

## Best Practices

1. **Start Small:** Begin with small models and datasets to verify setup
2. **Monitor Resources:** Keep an eye on RAM/GPU usage during experiments
3. **Use Version Control:** Track experiment configurations and results
4. **Reproducibility:** Use fixed seeds for comparable results
5. **Incremental Testing:** Test with few iterations before full runs
6. **Hardware Matching:** Choose models appropriate for your hardware
7. **Backup Results:** Save important experimental logs

## Getting Help

1. **Check this guide** for common issues and solutions
2. **Run verification script:** `python verify_ollama_integration.py`
3. **Test simple example:** `python examples/ollama_example.py`
4. **Check Ollama docs:** [https://ollama.ai/docs](https://ollama.ai/docs)
5. **Monitor system resources** during experiments

---

**Happy Experimenting!** 🚀

For more advanced usage and customization, refer to the source code in `src/experiments/ICM.py` and `src/experiments/ICM_tools.py`.
